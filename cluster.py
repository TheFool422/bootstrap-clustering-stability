import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[variable] = '1'

ROOT = Path(__file__).resolve().parent
FIELDS = ['phase','dataset','sample','representation','dimension','cluster_size',
          'radius_quantile','B','threshold','successes','S_score']


def settings():
    return json.loads((ROOT/'settings.json').read_text())


def read_array(source, member):
    import numpy as np
    source=Path(source)
    if source.is_dir():
        return np.load(source/member,allow_pickle=False)
    with zipfile.ZipFile(source) as z, z.open(member) as f:
        return np.load(io.BytesIO(f.read()),allow_pickle=False)


def source_hash(source, member):
    source=Path(source); h=hashlib.sha256()
    if source.is_dir():
        with (source/member).open('rb') as f:
            for block in iter(lambda:f.read(4<<20),b''): h.update(block)
    else:
        with zipfile.ZipFile(source) as z, z.open(member) as f:
            for block in iter(lambda:f.read(4<<20),b''): h.update(block)
    return h.hexdigest()


# Fit the recorded clustering rule, recalculating DBSCAN radii for each matrix.
def fit_labels(X, sizes, quantiles, dbscan):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    scaled=StandardScaler().fit_transform(X)
    if not dbscan:
        from hdbscan import HDBSCAN
        return [HDBSCAN(min_cluster_size=c,core_dist_n_jobs=1).fit_predict(scaled).astype(np.int32)
                for c in sizes]
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import DBSCAN
    distances,_=NearestNeighbors(n_neighbors=max(sizes),algorithm='brute',n_jobs=1).fit(scaled).kneighbors()
    labels=[]
    for c in sizes:
        core=distances[:,c-1]
        for q in quantiles:
            eps=float(np.quantile(core,q))
            if eps<=0:
                positive=core[core>0]; eps=float(positive.min()) if len(positive) else 1e-12
            labels.append(DBSCAN(eps=eps,min_samples=c+1,n_jobs=1).fit_predict(scaled).astype(np.int32))
    return labels


# Keep encoder outputs and reductions fixed while resampling feature columns.
def bootstrap(X, sizes, quantiles, dbscan, B):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score
    baseline=np.stack(fit_labels(X,sizes,quantiles,dbscan))
    ari=np.empty((len(baseline),B),np.float32)
    for b in range(B):
        indices=np.random.RandomState(b).choice(X.shape[1],size=X.shape[1],replace=True)
        labels=fit_labels(X[:,indices],sizes,quantiles,dbscan)
        ari[:,b]=[adjusted_rand_score(reference,replica) for reference,replica in zip(baseline,labels)]
    return baseline,ari


def versions():
    import platform
    from importlib.metadata import version, PackageNotFoundError
    result={'python':platform.python_version()}
    for package in ['numpy','scipy','scikit-learn','umap-learn','hdbscan','numba']:
        try: result[package]=version(package)
        except PackageNotFoundError: result[package]='unavailable'
    return result


# Compute a representation/dimension job and retain labels and replica ARIs for audit.
def run_job(job):
    import numpy as np
    phase,ds,sample,rep,dimension,source,output,sizes,quantiles,B,cfg=job
    prefix=f'{ds}/{sample}'
    model='bert' if phase=='P2' else 'bigbird'
    base=Path(output)/phase/ds/sample
    base.mkdir(parents=True,exist_ok=True)
    target=base/f'{rep}_d{dimension}.npz'
    members=[f'{prefix}/H.npy']
    if rep!='H': members.append(f'{prefix}/{model}/{rep[-1]}.npy')
    signature={'phase':phase,'representation':rep,'dimension':dimension,'sizes':sizes,
               'quantiles':quantiles,'B':B,'seed':cfg['seed'],'umap':cfg['umap'],
               'thresholds':cfg['thresholds'],'versions':versions(),
               'inputs':{m:source_hash(source,m) for m in members}}
    key=json.dumps(signature,sort_keys=True)
    if target.exists():
        with np.load(target,allow_pickle=False) as z:
            if str(z['signature'])!=key:
                raise ValueError(f'Cached settings differ for {target}; choose a new --output directory.')
            ari=z['ari'].copy()
    else:
        h=read_array(source,members[0]).astype(np.float64)
        start=time.perf_counter(); reduced=None
        if rep=='H': X=h
        else:
            embedding=read_array(source,members[1])
            if phase=='P3':
                from sklearn.decomposition import PCA
                z=PCA(n_components=max(cfg['compact_dimensions']),svd_solver='full',whiten=False).fit_transform(embedding.astype(np.float64))
                reduced=z[:,:dimension].astype(np.float32)
            else:
                from umap import UMAP
                reduced=UMAP(n_components=dimension,random_state=cfg['seed'],n_jobs=1,**cfg['umap']).fit_transform(embedding).astype(np.float32)
            if len(reduced)!=len(h): raise ValueError('Embedding and metadata row counts differ.')
            X=np.hstack((h,reduced.astype(np.float64))) if rep.startswith('H') else reduced.astype(np.float64)
        reduction_seconds=time.perf_counter()-start
        start=time.perf_counter()
        labels,ari=bootstrap(X,sizes,quantiles,phase=='P4',B)
        arrays={'baseline_labels':labels,'ari':ari,'signature':np.array(key),
                'reduction_seconds':reduction_seconds,'clustering_and_bootstrap_seconds':time.perf_counter()-start}
        if reduced is not None: arrays['reduced_features']=reduced
        temporary=target.with_suffix('.partial')
        with temporary.open('wb') as f: np.savez_compressed(f,**arrays)
        temporary.replace(target)
    cells=[(c,q) for c in sizes for q in quantiles] if phase=='P4' else [(c,'') for c in sizes]
    rows=[]
    for i,(c,q) in enumerate(cells):
        for threshold in cfg['thresholds']:
            k=int((ari[i]>=threshold).sum())
            rows.append([phase,ds,sample,rep,dimension,c,q,B,threshold,k,k/B])
    print(f'{phase} {ds} {sample} {rep} d={dimension}: complete',flush=True)
    return rows


def run(args):
    cfg=settings()
    cache=args.output.resolve()/'.numba_cache'
    cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('NUMBA_CACHE_DIR',str(cache))
    phases=args.phases or ['P0','P1','P2','P3','P4']
    datasets=args.dataset or list(cfg['samples'])
    B=args.replicas or cfg['bootstrap_replicas']
    if B<1: raise ValueError('Replica count must be positive.')
    if args.workers<1: raise ValueError('Worker count must be positive.')
    jobs=[]
    for phase in phases:
        windows=['W2','W3','W4'] if phase=='P1' else ['W1']
        if args.samples: windows=[w for w in windows if w in args.samples]
        dims=args.dimensions or cfg['broad_dimensions' if phase=='P0' else 'compact_dimensions']
        sizes=args.cluster_sizes or cfg['broad_cluster_sizes' if phase=='P0' else 'compact_cluster_sizes']
        if any(d<1 for d in dims) or any(c<2 for c in sizes): raise ValueError('Invalid grid.')
        if phase=='P3' and max(dims)>max(cfg['compact_dimensions']):
            raise ValueError('P3 dimensions must not exceed the saved PCA basis dimension.')
        for ds in datasets:
            for sample in windows:
                for rep in args.representations or ['H','F','A','HF','HA']:
                    for dim in [0] if rep=='H' else dims:
                        jobs.append((phase,ds,sample,rep,dim,str(args.embeddings.resolve()),
                                     str(args.output.resolve()),sizes,cfg['dbscan_quantiles'],B,cfg))
    if not jobs: raise ValueError('No jobs match the requested phase/sample combination.')
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    if args.workers==1:
        for job in jobs: results.extend(run_job(job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for rows in pool.map(run_job,jobs): results.extend(rows)
    path=args.output/'results.csv'
    temporary=path.with_suffix('.partial')
    with temporary.open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(FIELDS);writer.writerows(results)
    temporary.replace(path)
    print(f'New results for this invocation: {path}. Archived results.csv is unchanged.')


# Report means and matched F/A wins, ties, and losses from archived or new scores.
def summarize(path):
    groups=defaultdict(list); pairs=defaultdict(dict)
    with Path(path).open(newline='') as f:
        for r in csv.DictReader(f):
            t=float(r['threshold']); value=float(r['S_score'])
            groups[(r['phase'],r['dataset'],r['sample'],r['representation'],t)].append(value)
            if r['representation'] in ['F','A']:
                key=(r['phase'],r['dataset'],r['sample'],r['dimension'],r['cluster_size'],r['radius_quantile'],t)
                pairs[key][r['representation']]=int(r['successes'])/int(r['B'])
    print('phase,dataset,sample,representation,threshold,mean_S_percent,settings')
    for key,values in sorted(groups.items()):
        print(','.join(map(str,key))+f',{100*sum(values)/len(values):.3f},{len(values)}')
    totals=defaultdict(lambda:[0,0,0])
    for key,values in pairs.items():
        if set(values)!=set(['F','A']): continue
        delta=values['F']-values['A']; bucket=0 if delta>0 else 1 if delta==0 else 2
        totals[(key[0],key[-1])][bucket]+=1
    print('\nphase,threshold,F_higher,equal,F_lower,F_equal_or_higher_percent')
    for key,(win,tie,loss) in sorted(totals.items()):
        print(f'{key[0]},{key[1]},{win},{tie},{loss},{100*(win+tie)/(win+tie+loss):.3f}')


def verify():
    cfg=settings(); count=0
    for name,expected in cfg['files_sha256'].items():
        if ':' in name:
            archive,member=name.split(':',1); actual=source_hash(ROOT/archive,member)
        else:
            h=hashlib.sha256()
            with (ROOT/name).open('rb') as f:
                for block in iter(lambda:f.read(4<<20),b''): h.update(block)
            actual=h.hexdigest()
        if actual!=expected: raise ValueError(f'Checksum mismatch: {name}')
        count+=1
    print(f'Verified {count} archived files against source-file SHA-256 checksums.')


def costs():
    import numpy as np
    from decimal import Decimal, ROUND_HALF_UP
    cfg=settings()
    print('dataset,representation,mean_retained_bytes,mean_recorded_seconds_per_session,timing_samples')
    with zipfile.ZipFile(ROOT/'embeddings.zip') as z:
        for ds,windows in cfg['samples'].items():
            for rep in ['F','A']:
                sizes=[]
                for window in windows:
                    with z.open(f'{ds}/{window}/bigbird/index_{rep}.csv') as raw:
                        sizes.extend(int(r['bytes_used']) for r in csv.DictReader(io.TextIOWrapper(raw)))
                times=[Decimal(str(r[rep]))/10000 for r in cfg['recorded_embedding_seconds'][ds].values()]
                mean_time=(sum(times)/len(times)).quantize(Decimal('0.00001'),rounding=ROUND_HALF_UP)
                print(f'{ds},{rep},{np.mean(sizes):.1f},{mean_time},{len(times)}')
    print(cfg['recorded_timing_note'])


def main():
    ap=argparse.ArgumentParser(description='Reduction, baseline clustering, bootstrap evaluation, and archived reports.')
    sub=ap.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run')
    r.add_argument('--phases',nargs='+',choices=['P0','P1','P2','P3','P4'])
    r.add_argument('--dataset',nargs='+',choices=['unsw_nb15','cicids2017'])
    r.add_argument('--samples',nargs='+',choices=['W1','W2','W3','W4'])
    r.add_argument('--representations',nargs='+',choices=['H','F','A','HF','HA'])
    r.add_argument('--dimensions',nargs='+',type=int)
    r.add_argument('--cluster-sizes',nargs='+',type=int)
    r.add_argument('--replicas',type=int)
    r.add_argument('--workers',type=int,default=1)
    r.add_argument('--embeddings',type=Path,default=ROOT/'embeddings.zip')
    r.add_argument('--output',type=Path,default=ROOT/'generated/clustering')
    s=sub.add_parser('summarize');s.add_argument('--results',type=Path,default=ROOT/'results.csv')
    sub.add_parser('verify');sub.add_parser('costs')
    a=ap.parse_args()
    if a.command=='run': run(a)
    elif a.command=='summarize': summarize(a.results)
    elif a.command=='verify': verify()
    elif a.command=='costs': costs()


if __name__=='__main__':
    main()
