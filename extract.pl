#!/usr/bin/perl

package ExtractCommon;
use strict;
use warnings;
use File::Spec;
use File::Basename qw(dirname);
use File::Path qw(make_path);
use Cwd qw(abs_path);
use POSIX qw(strftime);
use Digest::SHA;
use Exporter 'import';
our @EXPORT_OK = qw(root_dir read_config cfg_path cfg_list open_log logmsg
                    write_counts read_counts sha256_file ensure_dir now_str);

my $ROOT;
my $LOG_FH;

sub root_dir {
    return $ROOT if defined $ROOT;
    $ROOT = abs_path(dirname(__FILE__));
    return $ROOT;
}

sub now_str { return strftime('%Y-%m-%d %H:%M:%S', localtime) }

# Read extraction settings and validate the required configuration keys.
sub read_config {
    my ($path) = @_;
    open(my $fh, '<', $path) or die "Cannot open config $path: $!\n";
    my %c;
    while (my $line = <$fh>) {
        $line =~ s/\r?\n$//;
        $line =~ s/\s+#.*$//;
        next if $line =~ /^\s*(#|$)/;
        my ($k, $v) = $line =~ /^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$/
            or die "Bad config line in $path: $line\n";
        $c{$k} = $v;
    }
    close $fh;
    $c{_CONFIG_PATH} = abs_path($path);
    $c{MAX_PACKETS}       //= 0;
    $c{LABEL_FORMAT}      //= 'unsw';
    $c{LABEL_TIME_RES}    //= 1;
    $c{LABEL_TZ_OFFSET}   //= 0;
    $c{LABEL_PM_BELOW}    //= 0;
    die "Config $path: LABEL_FORMAT must be unsw or cicids\n" unless $c{LABEL_FORMAT} =~ /^(unsw|cicids)$/;
    for my $req (qw(DATASET SAMPLE PCAP_DIR PCAP_FILES LABEL_FILES TSHARK N CANDIDATE_MARGIN
                    IDLE_TIMEOUT STOP_BUFFER A_CAP_BYTES F_CUT_LEN LABEL_TOL OUT_DIR TMP_DIR LOG_DIR)) {
        die "Config $path: missing key $req\n" unless defined $c{$req} && length $c{$req};
    }
    return \%c;
}

# Resolve a configured path relative to the repository root.
sub cfg_path {
    my ($c, $key) = @_;
    my $p = $c->{$key};
    return $p if File::Spec->file_name_is_absolute($p);
    return File::Spec->catfile(root_dir(), $p);
}

sub cfg_list {
    my ($c, $key) = @_;
    return grep { length } map { s/^\s+|\s+$//gr } split /,/, $c->{$key};
}

sub ensure_dir { my ($d) = @_; make_path($d) unless -d $d; return $d }

sub open_log {
    my ($c, $script) = @_;
    my $dir = ensure_dir(cfg_path($c, 'LOG_DIR'));
    my $day = strftime('%Y-%m-%d', localtime);
    my $f = File::Spec->catfile($dir, "${day}_$c->{DATASET}_$c->{SAMPLE}_01_extract.log");
    open($LOG_FH, '>>', $f) or die "Cannot open log $f: $!\n";
    my $old = select($LOG_FH); $| = 1; select($old);
    logmsg("===== $script started (config: $c->{_CONFIG_PATH}) =====");
    return $f;
}

sub logmsg {
    my ($msg) = @_;
    my $line = '[' . now_str() . "] $msg\n";
    print STDOUT $line;
    print $LOG_FH $line if $LOG_FH;
}

sub write_counts {
    my ($path, $pairs) = @_;
    open(my $fh, '>', $path) or die "Cannot write $path: $!\n";
    printf $fh "%s = %s\n", $_->[0], $_->[1] for @$pairs;
    close $fh;
}

sub read_counts {
    my ($path) = @_;
    my @pairs;
    return @pairs unless -e $path;
    open(my $fh, '<', $path) or die "Cannot read $path: $!\n";
    while (<$fh>) { s/\r?\n$//; push @pairs, [$1, $2] if /^(\S+) = (.*)$/ }
    close $fh;
    return @pairs;
}

sub sha256_file {
    my ($f) = @_;
    my $sha = Digest::SHA->new(256);
    $sha->addfile($f, 'b');
    return $sha->hexdigest;
}

1;


package Stage_packets;
sub run {
use strict;
use warnings;
no warnings "closure";
use File::Spec;
use File::Basename qw(basename);
BEGIN {
    *read_config = \&ExtractCommon::read_config;
    *cfg_path = \&ExtractCommon::cfg_path;
    *cfg_list = \&ExtractCommon::cfg_list;
    *open_log = \&ExtractCommon::open_log;
    *logmsg = \&ExtractCommon::logmsg;
    *write_counts = \&ExtractCommon::write_counts;
    *sha256_file = \&ExtractCommon::sha256_file;
    *ensure_dir = \&ExtractCommon::ensure_dir;
}


my $cfg_file = shift @ARGV or die "Usage: perl s1_packets.pl <config> [--force]\n";
my $force = grep { $_ eq '--force' } @ARGV;
my $c = read_config($cfg_file);
open_log($c, 's1_packets.pl');

my $tmp = ensure_dir(cfg_path($c, 'TMP_DIR'));
my $pcap_dir = cfg_path($c, 'PCAP_DIR');
my $tshark = $c->{TSHARK};

# Read packet-dissection output from a subprocess without a shell.
sub run_lines {
    my @cmd = @_;
    open(my $p, '-|', @cmd) or die "Cannot run $cmd[0]: $!\n";
    my @l = <$p>; close $p;
    return join '', @l;
}
my $ver = eval { run_lines($tshark, '-v') };
die "Cannot run tshark ($tshark). Set TSHARK in the config.\n" unless defined $ver && $ver =~ /TShark/i;
my ($ver_line) = split /\n/, $ver;
logmsg("tshark: $ver_line");
logmsg("perl: $^V");

my @fields = qw(frame.number frame.time_epoch ip.src ip.dst ip.id tcp.srcport tcp.dstport
                tcp.flags tcp.seq tcp.ack tcp.len tcp.payload tcp.options.mss_val tcp.options.timestamp.tsval);
my $known = eval { run_lines($tshark, '-G', 'fields') } // '';
for my $fld (qw(tcp.options.mss_val tcp.options.timestamp.tsval)) {
    die "This tshark has no field $fld - update Wireshark (4.x has it).\n" unless $known =~ /\t\Q$fld\E\t/;
}
my $SCHEMA = 2;
my @opts = (
    '-n',
    '-o', 'tcp.desegment_tcp_streams:FALSE',
    '-o', 'tcp.relative_sequence_numbers:FALSE',
    '-o', 'tcp.analyze_sequence_numbers:FALSE',
    '-Y', 'ip && tcp && !icmp',
    '-T', 'fields', '-E', 'separator=/t', '-E', 'occurrence=f', '-E', 'quote=n', '-E', 'header=n',
    map { ('-e', $_) } @fields,
);

push @opts, ('-c', $c->{MAX_PACKETS}) if $c->{MAX_PACKETS} > 0;
logmsg("MAX_PACKETS = $c->{MAX_PACKETS}" . ($c->{MAX_PACKETS} > 0 ? '' : ' (whole file)'));
my @counts = ([tshark_version => $ver_line], [max_packets => $c->{MAX_PACKETS}]);
for my $pc (cfg_list($c, 'PCAP_FILES')) {
    my $pcap = File::Spec->file_name_is_absolute($pc) ? $pc : File::Spec->catfile($pcap_dir, $pc);
    die "Missing pcap $pcap\n" unless -e $pcap;
    (my $name = basename($pc)) =~ s/\.(pcap|pcapng)$//i;
    my $out = File::Spec->catfile($tmp, "pkts_$name.tsv");

    my $meta = "$out.meta";
    my $meta_ok = 0;
    if (-e $meta) {
        open(my $m, '<', $meta) or die; local $/; my $v = <$m>; close $m;
        $meta_ok = defined $v && $v =~ /max_packets = (\d+)/ && $1 == $c->{MAX_PACKETS} && $v =~ /schema = $SCHEMA\b/;
    }
    logmsg("$pc: existing packet table is missing, has another MAX_PACKETS, or is an old format -> rebuilding") if !$force && -e $out && !$meta_ok;
    if (!$force && $meta_ok && -e $out && (stat $out)[9] >= (stat $pcap)[9] && -s $out) {
        logmsg("$pc: $out exists and is up to date, skipped (use --force to redo)");
        my $n = 0; open(my $f, '<', $out) or die; while (<$f>) { $n++ unless /^#/ } close $f;
        push @counts, ["pkts_rows_$name" => $n];
        next;
    }

    logmsg("$pc: sha256 ...");
    my $sha = sha256_file($pcap);
    logmsg("$pc: sha256 = $sha");
    push @counts, ["sha256_$name" => $sha];

    my @cmd = ($tshark, '-r', $pcap, @opts);
    logmsg("$pc: " . join(' ', map { /\s/ ? qq{"$_"} : $_ } @cmd));
    open(my $ts, '-|', @cmd) or die "Cannot run tshark: $!\n";
    open(my $fo, '>', "$out.part") or die "Cannot write $out.part: $!\n";
    binmode $fo;
    print $fo "#frame_number\tt_epoch\tip_src\tip_dst\tip_id\ttcp_srcport\ttcp_dstport\ttcp_flags\ttcp_seq_raw\ttcp_ack_raw\ttcp_len\ttcp_payload\ttcp_mss\ttcp_ts_opt\n";
    my ($n, $t0) = (0, time);
    while (my $line = <$ts>) {
        $line =~ s/\r?\n$//;
        my @f = split /\t/, $line, -1;
        next unless @f >= 11 && length $f[2] && length $f[5];
        $f[11] = '' unless defined $f[11];
        $f[11] =~ s/://g;
        $f[11] = lc $f[11];
        my ($mss, $tso) = ('', '');
        if (hex($f[7]) & 0x02) {
            $mss = defined $f[12] ? $f[12] : '';
            $tso = (defined $f[13] && length $f[13]) ? 1 : 0;
        }
        print $fo join("\t", @f[0..11], $mss, $tso), "\n";
        $n++;
        logmsg(sprintf("  %s: %d packets (%ds)", $pc, $n, time - $t0)) if $n % 1_000_000 == 0;
    }
    close $ts;
    my $rc = $? >> 8;
    close $fo;
    die "tshark failed on $pc (exit $rc)\n" if $rc != 0;
    unlink $out if -e $out;
    rename("$out.part", $out) or die "Cannot rename $out.part: $!\n";
    open(my $m, '>', $meta) or die "Cannot write $meta: $!\n"; print $m "max_packets = $c->{MAX_PACKETS}\nschema = $SCHEMA\n"; close $m;
    logmsg(sprintf("%s: %d TCP packets written to %s (%ds)", $pc, $n, $out, time - $t0));
    push @counts, ["pkts_rows_$name" => $n];
}

write_counts(File::Spec->catfile($tmp, 'counts_s1.txt'), \@counts);
logmsg('s1_packets.pl done');

}


package Stage_sessions;
sub run {
use strict;
use warnings;
no warnings "closure";
use File::Spec;
use File::Basename qw(basename);
use POSIX qw(ceil);
BEGIN {
    *read_config = \&ExtractCommon::read_config;
    *cfg_path = \&ExtractCommon::cfg_path;
    *cfg_list = \&ExtractCommon::cfg_list;
    *open_log = \&ExtractCommon::open_log;
    *logmsg = \&ExtractCommon::logmsg;
    *write_counts = \&ExtractCommon::write_counts;
    *ensure_dir = \&ExtractCommon::ensure_dir;
}


use constant { FIN => 0x01, SYN => 0x02, RST => 0x04, ACK => 0x10 };
my $MOD  = 4294967296;
my $HALF = 2147483648;

my $cfg_file = shift @ARGV or die "Usage: perl s2_sessions.pl <config>\n";
my $c = read_config($cfg_file);
open_log($c, 's2_sessions.pl');

my $tmp    = ensure_dir(cfg_path($c, 'TMP_DIR'));
my $target = ceil($c->{N} * (1 + $c->{CANDIDATE_MARGIN}));
my ($IDLE, $BUF, $CAP, $FCUT) = @$c{qw(IDLE_TIMEOUT STOP_BUFFER A_CAP_BYTES F_CUT_LEN)};
logmsg("candidate target = $target (N=$c->{N}, margin=$c->{CANDIDATE_MARGIN})");

my @CAND_COLS = qw(cand_id pcap_first frame_first pcap_last client_ip client_port server_ip server_port
                   t_syn t_synack t_ack respond_time t_end duration end_reason
                   c2s_pkts s2c_pkts c2s_payload_bytes s2c_payload_bytes
                   dup_frames_dropped retrans_dropped keepalive_dropped c2s_gap_bytes
                   seg_limit f_raw_len f_len f_hex f_merged f_maybe_cut a_len_total a_cap_len a_cap_hex a_truncated);

open(my $CAND, '>', File::Spec->catfile($tmp, 'candidates.tsv.part')) or die "$!\n";
open(my $AFUL, '>', File::Spec->catfile($tmp, 'a_full.tsv.part'))     or die "$!\n";
binmode $CAND; binmode $AFUL;
print $CAND join("\t", @CAND_COLS), "\n";
print $AFUL "cand_id\ta_len_total\ta_full_hex\n";

my %C = map { $_ => 0 } qw(packets_read dup_frames_dropped retrans_dropped keepalive_dropped sessions_opened
    handshake_incomplete excluded_no_payload qualified_written orphan_packets_no_handshake
    reused_port_new_session packets_after_stop_untracked client_seg_before_isn);
my (%sess, %orphan_keys);
my ($next_id, $qualified, $stop_opening, $stop_time, $last_ts) = (0, 0, 0, undef, '');
my @used;

FILE: for my $pc (cfg_list($c, 'PCAP_FILES')) {
    (my $name = basename($pc)) =~ s/\.(pcap|pcapng)$//i;
    my $in = File::Spec->catfile($tmp, "pkts_$name.tsv");
    die "Missing $in - run s1_packets.pl first\n" unless -e $in;
    push @used, $pc;
    logmsg("reading $in");
    open(my $fh, '<', $in) or die "$!\n";
    { my $h = <$fh>; die "$in has no tcp_mss / tcp_ts_opt columns (old format) - rerun s1_packets.pl\n" unless defined $h && $h =~ /tcp_mss\ttcp_ts_opt/; }
    while (my $line = <$fh>) {
        next if substr($line, 0, 1) eq '#';
        $line =~ s/\r?\n$//;
        my ($fr, $ts, $src, $dst, $ipid, $sp, $dp, $flx, $seq, $ack, $len, $pl, $mss, $tso) = split /\t/, $line, -1;
        $C{packets_read}++;
        my $t = $ts + 0;
        $last_ts = $ts;
        last FILE if $stop_opening && $t > $stop_time + $BUF;
        logmsg("  $C{packets_read} packets, $next_id sessions opened, $qualified qualified")
            if $C{packets_read} % 1_000_000 == 0;

        my $fl = hex($flx);
        $len = 0 unless length $len;
        $pl  = '' unless defined $pl;
        my ($ea, $eb) = ("$src:$sp", "$dst:$dp");
        my $key = ($ea lt $eb) ? "$ea|$eb" : "$eb|$ea";
        my $s = $sess{$key};

        if ($s && $t - $s->{t_last_n} > $IDLE) {
            finalize($s, 'timeout'); delete $sess{$key}; $s = undef;
        }

        if (($fl & SYN) && !($fl & ACK)) {
            if ($s && !($s->{cip} eq $src && $s->{cport} == $sp && $s->{isn_c} == $seq)) {
                finalize($s, 'reused'); delete $sess{$key}; $s = undef;
                $C{reused_port_new_session}++;
            }
            if (!$s) {
                if ($stop_opening) { $C{packets_after_stop_untracked}++; next }
                $s = { id => ++$next_id, cip => $src, cport => $sp, sip => $dst, sport => $dp,
                       mss_c => $mss, ts_c => $tso, isn_c => $seq, t_syn => $ts, t_syn_n => $t, pcap_first => $pc, frame_first => $fr,
                       c2s_pkts => 0, s2c_pkts => 0, c2s_bytes => 0, s2c_bytes => 0,
                       dup => 0, retrans => 0, keepalive => 0, maxend => 0, segs => {}, last_sig => {} };
                $sess{$key} = $s;
                $C{sessions_opened}++;
            }
        }
        if (!$s) {
            if ($stop_opening) { $C{packets_after_stop_untracked}++ }
            else { $C{orphan_packets_no_handshake}++; $orphan_keys{$key} = 1 }
            next;
        }

        my $dir = ($src eq $s->{cip} && $sp == $s->{cport}) ? 'c' : 's';

        my $sig = "$ipid|$fl|$seq|$ack|$len|$pl";
        if (defined $s->{last_sig}{$dir} && $s->{last_sig}{$dir} eq $sig) {
            $s->{dup}++; $C{dup_frames_dropped}++; next;
        }
        $s->{last_sig}{$dir} = $sig;

        $s->{t_last_n} = $t; $s->{t_last} = $ts; $s->{pcap_last} = $pc;
        $s->{$dir eq 'c' ? 'c2s_pkts' : 's2c_pkts'}++;

        if ($fl & RST) { $s->{closed} = 1; $s->{end_reason} //= 'rst' }
        if ($fl & FIN) {
            $s->{"fin_$dir"} = 1;
            if ($s->{fin_c} && $s->{fin_s} && !$s->{closed}) { $s->{closed} = 1; $s->{end_reason} = 'fin' }
        }

        if ($dir eq 's' && ($fl & SYN) && ($fl & ACK)) {
            if (!defined $s->{t_synack}) { $s->{t_synack} = $ts; $s->{t_synack_n} = $t; $s->{mss_s} = $mss; $s->{ts_s} = $tso }
        } elsif ($dir eq 'c' && ($fl & ACK) && !($fl & SYN) && defined $s->{t_synack} && !defined $s->{t_ack}) {
            $s->{t_ack} = $ts; $s->{t_ack_n} = $t; $s->{hs_ok} = 1;
        }

        next unless $len > 0;
        if ($dir eq 's') { $s->{s2c_bytes} += $len; next }

        my $off = ($seq - $s->{isn_c} - 1) % $MOD;
        if ($off >= $HALF) { $C{client_seg_before_isn}++; next }
        my $bin = pack('H*', $pl);
        my $ex  = $s->{segs}{$off};
        if (length($bin) == 1 && $s->{maxend} > 0 && $off == $s->{maxend} - 1) { $s->{keepalive}++; $C{keepalive_dropped}++; next }
        if (defined $ex && length($ex) >= length($bin)) { $s->{retrans}++; $C{retrans_dropped}++; next }
        $s->{c2s_bytes} += length($bin) - (defined $ex ? length($ex) : 0);
        if (defined $ex) { $s->{orig}{$off} //= $ex; $C{longer_copy_same_offset}++ }
        $s->{segs}{$off} = $bin;
        { my $end = $off + length $bin; $s->{maxend} = $end if $end > $s->{maxend}; }

        if ($s->{hs_ok} && !$s->{qualified}) {
            $s->{qualified} = 1;
            if (++$qualified >= $target && !$stop_opening) {
                $stop_opening = 1; $stop_time = $t;
                logmsg("candidate target reached at t=$ts (packet $C{packets_read}); no new sessions, reading $BUF s more");
            }
        }
    }
    close $fh;
}

finalize($_, 'capture_end') for sort { $a->{id} <=> $b->{id} } values %sess;
%sess = ();
close $CAND; close $AFUL;
for my $f (qw(candidates.tsv a_full.tsv)) {
    my $p = File::Spec->catfile($tmp, $f);
    unlink $p if -e $p;
    rename("$p.part", $p) or die "Cannot rename $p.part: $!\n";
}

my $enough = $qualified >= $target ? 'yes' : 'NO - add the next pcap to PCAP_FILES and rerun';
logmsg("qualified sessions: $qualified (target $target) enough=$enough");
write_counts(File::Spec->catfile($tmp, 'counts_s2.txt'), [
    [pcaps_used => join(',', @used)],
    [packets_read => $C{packets_read}],
    [last_packet_time => $last_ts],
    [candidate_target => $target],
    [candidate_target_reached => $enough],
    [stop_opening_at => defined $stop_time ? $stop_time : 'never'],
    [sessions_opened => $C{sessions_opened}],
    [excluded_handshake_incomplete => $C{handshake_incomplete}],
    [excluded_no_payload => $C{excluded_no_payload}],
    [excluded_no_handshake_4tuples => scalar(keys %orphan_keys)],
    [excluded_no_handshake_packets => $C{orphan_packets_no_handshake}],
    [qualified_written => $C{qualified_written}],
    [dup_frames_dropped => $C{dup_frames_dropped}],
    [retrans_dropped => $C{retrans_dropped}],
    [keepalive_dropped => $C{keepalive_dropped}],
    [reused_port_new_session => $C{reused_port_new_session}],
    [client_seg_before_isn => $C{client_seg_before_isn}],
    [longer_copy_same_offset => $C{longer_copy_same_offset} // 0],
    [f_merged_cut_to_segment_limit => $C{f_merged} // 0],
    [seg_limit_from_default_mss => $C{seg_limit_default_mss} // 0],
    (map { [$_ => $C{$_}] } sort grep { /^seg_limit_\d+$/ } keys %C),
    [packets_after_stop_untracked => $C{packets_after_stop_untracked}],
]);
logmsg('s2_sessions.pl done');

# Reassemble a session and derive the first and all client payloads.
sub finalize {
    my ($s, $reason) = @_;
    if (!$s->{hs_ok})     { $C{handshake_incomplete}++; return }
    if (!$s->{qualified}) { $C{excluded_no_payload}++;  return }

    my @offs = sort { $a <=> $b } keys %{ $s->{segs} };
    my ($stream, $cur, $gap) = ('', 0, 0);
    for my $o (@offs) {
        my $d = $s->{segs}{$o};
        my $e = $o + length $d;
        if    ($o > $cur) { $gap += $o - $cur; $stream .= $d; $cur = $e }
        elsif ($e > $cur) { $stream .= substr($d, $cur - $o); $cur = $e }
    }
    my $f_raw = exists $s->{orig}{ $offs[0] } ? $s->{orig}{ $offs[0] } : $s->{segs}{ $offs[0] };
    my @mss = grep { defined && length } ($s->{mss_c}, $s->{mss_s});
    my $mss = @mss ? (sort { $a <=> $b } @mss)[0] : $FCUT;
    $C{seg_limit_default_mss}++ unless @mss;
    my $ts_on = (defined $s->{ts_c} && $s->{ts_c} eq '1' && defined $s->{ts_s} && $s->{ts_s} eq '1') ? 1 : 0;
    my $limit = $mss - ($ts_on ? 12 : 0);
    $limit = $FCUT if $limit < 1;
    $C{"seg_limit_$limit"}++;
    my $merged = length($f_raw) > $limit ? 1 : 0;
    $C{f_merged}++ if $merged;
    my $f = $merged ? substr($f_raw, 0, $limit) : $f_raw;
    my $alen = length $stream;
    my $acap = $alen > $CAP ? substr($stream, 0, $CAP) : $stream;
    my %r = (
        cand_id => $s->{id}, pcap_first => $s->{pcap_first}, frame_first => $s->{frame_first},
        pcap_last => $s->{pcap_last}, client_ip => $s->{cip}, client_port => $s->{cport},
        server_ip => $s->{sip}, server_port => $s->{sport},
        t_syn => $s->{t_syn}, t_synack => $s->{t_synack}, t_ack => $s->{t_ack},
        respond_time => sprintf('%.6f', $s->{t_ack_n} - $s->{t_synack_n}),
        t_end => $s->{t_last}, duration => sprintf('%.6f', $s->{t_last_n} - $s->{t_syn_n}),
        end_reason => ($s->{closed} ? $s->{end_reason} : $reason),
        c2s_pkts => $s->{c2s_pkts}, s2c_pkts => $s->{s2c_pkts},
        c2s_payload_bytes => $s->{c2s_bytes}, s2c_payload_bytes => $s->{s2c_bytes},
        dup_frames_dropped => $s->{dup}, retrans_dropped => $s->{retrans}, keepalive_dropped => $s->{keepalive}, c2s_gap_bytes => $gap,
        seg_limit => $limit, f_raw_len => length $f_raw,
        f_len => length $f, f_hex => unpack('H*', $f), f_merged => $merged,
        f_maybe_cut => (length($f) == $limit ? 1 : 0),
        a_len_total => $alen, a_cap_len => length $acap, a_cap_hex => unpack('H*', $acap),
        a_truncated => ($alen > $CAP ? 1 : 0),
    );
    print $CAND join("\t", map { defined $r{$_} ? $r{$_} : '' } @CAND_COLS), "\n";
    print $AFUL "$s->{id}\t$alen\t", unpack('H*', $stream), "\n";
    $C{qualified_written}++;
    $s->{segs} = undef; $s->{orig} = undef;
}

}


package Stage_labels;
sub run {
use strict;
use warnings;
no warnings "closure";
use File::Spec;
use POSIX qw(floor);
use Time::Local qw(timegm);
BEGIN {
    *read_config = \&ExtractCommon::read_config;
    *cfg_path = \&ExtractCommon::cfg_path;
    *cfg_list = \&ExtractCommon::cfg_list;
    *open_log = \&ExtractCommon::open_log;
    *logmsg = \&ExtractCommon::logmsg;
    *write_counts = \&ExtractCommon::write_counts;
}


my $cfg_file = shift @ARGV or die "Usage: perl s4_labels.pl <config>\n";
my $c = read_config($cfg_file);
open_log($c, 's4_labels.pl');
my $tmp = cfg_path($c, 'TMP_DIR');
my ($TOL, $RES, $TZ, $PM, $FMT) = @$c{qw(LABEL_TOL LABEL_TIME_RES LABEL_TZ_OFFSET LABEL_PM_BELOW LABEL_FORMAT)};
logmsg("label format $FMT, time resolution $RES s, tolerance $TOL s, tz offset $TZ s");

my $in = File::Spec->catfile($tmp, 'candidates.tsv');
open(my $fh, '<', $in) or die "Missing $in - run s2_sessions.pl first\n";
my $hdr = <$fh>; $hdr =~ s/\r?\n$//;
my @cols = split /\t/, $hdr;
my %ix; @ix{@cols} = 0 .. $#cols;
my %want;
my $n_cand = 0;
while (my $l = <$fh>) {
    $l =~ s/\r?\n$//;
    my @f = split /\t/, $l, -1;
    $want{ join(',', @f[@ix{qw(client_ip client_port server_ip server_port)}]) } = 1;
    $n_cand++;
}
close $fh;
logmsg("candidates: $n_cand, distinct 4-tuples: " . scalar(keys %want));

my (%rows, %seen);
my ($tcp_rows, $matched_rows, $dup_rows) = (0, 0, 0);
for my $lf (cfg_list($c, 'LABEL_FILES')) {
    my $p = File::Spec->file_name_is_absolute($lf) ? $lf : File::Spec->catfile(ExtractCommon::root_dir(), $lf);
    open(my $lh, '<', $p) or die "Cannot open label file $p: $!\n";
    logmsg("reading $p");
    my $first = 1;
    while (my $l = <$lh>) {
        if ($first) {
            $first = 0;
            $l =~ s/^\x{EF}\x{BB}\x{BF}//;
            next if $FMT eq 'cicids';
        }
        $l =~ s/\r?\n$//;
        my @f = split /,/, $l, -1;
        my ($k, $start, $end, $cat, $lab);
        if ($FMT eq 'unsw') {
            next unless @f >= 49;
            $_ =~ s/^\s+|\s+$//g for @f[0 .. 4, 28, 29, 47, 48];
            next unless $f[4] eq 'tcp';
            $k = join(',', @f[0 .. 3]);
            ($start, $end, $lab) = ($f[28], $f[29], $f[48]);
            $cat = $f[47];
            $cat = 'Backdoor' if $cat eq 'Backdoors';
            $cat = 'Normal'   if $lab eq '0';
        } else {
            next unless @f >= 85;
            $_ =~ s/^\s+|\s+$//g for @f[1 .. 6, 84];
            next unless $f[5] eq '6';
            $k = join(',', @f[1 .. 4]);
            $start = cicids_time($f[6]);
            next unless defined $start;
            $end = '';
            if ($f[84] eq 'BENIGN') { ($lab, $cat) = ('0', 'Normal') } else { ($lab, $cat) = ('1', $f[84]) }
        }
        $tcp_rows++;
        next unless $want{$k};
        if ($seen{$l}++) { $dup_rows++; next }
        $matched_rows++;
        push @{ $rows{$k} }, [ $start, $end, $cat, $lab ];
    }
    close $lh;
}
%seen = ();
logmsg("label TCP rows: $tcp_rows; rows on candidate 4-tuples: $matched_rows; exact duplicates removed: $dup_rows");

my %status;
open($fh, '<', $in) or die "$!\n";
<$fh>;
my $out = File::Spec->catfile($tmp, 'candidates_labeled.tsv');
open(my $fo, '>', "$out.part") or die "$!\n";
binmode $fo;
print $fo join("\t", @cols, qw(label attack_cat label_stime label_ltime match_status n_label_rows)), "\n";
while (my $l = <$fh>) {
    $l =~ s/\r?\n$//;
    my @f = split /\t/, $l, -1;
    my $k = join(',', @f[@ix{qw(client_ip client_port server_ip server_port)}]);
    my $ts = floor($f[ $ix{t_syn} ]);
    my $tsyn = $f[ $ix{t_syn} ];
    my @cand = grep { $tsyn >= $_->[0] - $TOL && $tsyn < $_->[0] + $RES + $TOL } @{ $rows{$k} || [] };
    my ($lab, $cat, $st, $lt, $ms) = ('', '', '', '', 'unlabeled');
    if (@cand) {
        my %cats = map { $_->[2] => 1 } @cand;
        if (keys(%cats) > 1) {
            $ms = 'multi_label';
        } else {
            my ($best) = sort { abs($a->[0] - $ts) <=> abs($b->[0] - $ts) } @cand;
            ($st, $lt, $cat, $lab) = @$best;
            $ms = @cand == 1 ? 'unique' : 'single_category';
        }
    }
    $status{$ms}++;
    print $fo join("\t", @f, $lab, $cat, $st, $lt, $ms, scalar(@cand)), "\n";
}
close $fh; close $fo;
unlink $out if -e $out;
rename("$out.part", $out) or die "$!\n";

logmsg(join(', ', map { "$_=" . ($status{$_} || 0) } qw(unique single_category multi_label unlabeled)));
write_counts(File::Spec->catfile($tmp, 'counts_s4.txt'), [
    [label_tcp_rows_read => $tcp_rows],
    [label_rows_on_candidate_4tuples => $matched_rows],
    [label_exact_duplicates_removed => $dup_rows],
    [match_unique => $status{unique} || 0],
    [match_single_category => $status{single_category} || 0],
    [dropped_multi_label => $status{multi_label} || 0],
    [dropped_unlabeled => $status{unlabeled} || 0],
]);
logmsg('s4_labels.pl done');

my $bad_ts = 0;
# Convert the benchmark timestamp into UTC for session-label matching.
sub cicids_time {
    my ($t) = @_;
    my ($d, $mo, $y, $h, $mi, $se) = $t =~ m{^(\d{1,2})/(\d{1,2})/(\d{4}) (\d{1,2}):(\d{2})(?::(\d{2}))?} or do {
        logmsg("WARNING: unparsed timestamp '$t'") if $bad_ts++ < 5; return undef };
    $h += 12 if $PM && $h < $PM;
    return timegm($se // 0, $mi, $h, $d, $mo - 1, $y) + $TZ;
}

}


package Stage_select;
sub run {
use strict;
use warnings;
no warnings "closure";
use File::Spec;
use File::Basename qw(basename);
use File::Copy qw(copy);
BEGIN {
    *read_config = \&ExtractCommon::read_config;
    *cfg_path = \&ExtractCommon::cfg_path;
    *open_log = \&ExtractCommon::open_log;
    *logmsg = \&ExtractCommon::logmsg;
    *read_counts = \&ExtractCommon::read_counts;
    *ensure_dir = \&ExtractCommon::ensure_dir;
    *now_str = \&ExtractCommon::now_str;
}


my $cfg_file = shift @ARGV or die "Usage: perl s5_select.pl <config>\n";
my $c = read_config($cfg_file);
open_log($c, 's5_select.pl');
my $tmp = cfg_path($c, 'TMP_DIR');
my $out = ensure_dir(cfg_path($c, 'OUT_DIR'));
my $N   = $c->{N};

my @OUT_COLS = qw(session_id pcap_first frame_first pcap_last client_ip client_port server_ip server_port
    t_syn t_synack t_ack respond_time t_end duration end_reason
    c2s_pkts s2c_pkts c2s_payload_bytes s2c_payload_bytes dup_frames_dropped retrans_dropped keepalive_dropped c2s_gap_bytes
    seg_limit f_raw_len f_len f_hex f_merged f_maybe_cut a_len_total a_cap_len a_cap_hex a_truncated
    label attack_cat label_stime label_ltime match_status);

my $in = File::Spec->catfile($tmp, 'candidates_labeled.tsv');
open(my $fh, '<', $in) or die "Missing $in - run s4_labels.pl first\n";
my $h = <$fh>; $h =~ s/\r?\n$//;
my @cols = split /\t/, $h;
my %ix; @ix{@cols} = 0 .. $#cols;
my @keep;
while (my $l = <$fh>) {
    $l =~ s/\r?\n$//;
    my @f = split /\t/, $l, -1;
    my $ms = $f[ $ix{match_status} ];
    push @keep, \@f if $ms eq 'unique' || $ms eq 'single_category';
}
close $fh;
@keep = sort { $a->[ $ix{t_syn} ] <=> $b->[ $ix{t_syn} ] or $a->[ $ix{cand_id} ] <=> $b->[ $ix{cand_id} ] } @keep;
my $available = @keep;
splice(@keep, $N) if @keep > $N;
my $selected = @keep;
logmsg("usable candidates: $available; selected: $selected (N=$N)");
logmsg("WARNING: fewer than N sessions - add the next pcap to PCAP_FILES and rerun s1..s5") if $selected < $N;

my $csv = File::Spec->catfile($out, "sessions_$c->{SAMPLE}.csv");
open(my $fo, '>', $csv) or die "$!\n";
binmode $fo;
print $fo join(',', @OUT_COLS), "\n";
my (%sid_of, %cat, %lab, %chk);
my $sid = 0;
for my $f (@keep) {
    $sid++;
    $sid_of{ $f->[ $ix{cand_id} ] } = $sid;
    my %r; @r{@cols} = @$f; $r{session_id} = $sid;
    print $fo join(',', map { defined $r{$_} ? $r{$_} : '' } @OUT_COLS), "\n";
    $cat{ $r{attack_cat} }++; $lab{ $r{label} }++;
    $chk{single_segment_a_eq_f}++ if $r{a_len_total} == $r{f_len};
    $chk{f_maybe_cut}++           if $r{f_maybe_cut};
    $chk{f_merged}++              if $r{f_merged};
    $chk{"seg_limit_$r{seg_limit}"}++;
    $chk{a_truncated}++           if $r{a_truncated};
    $chk{with_gap}++              if $r{c2s_gap_bytes} > 0;
    $chk{sessions_with_dups}++    if $r{dup_frames_dropped} > 0;
    $chk{"end_$r{end_reason}"}++;
    $chk{respond_time_gt_5s}++    if $r{respond_time} > 5;
}
close $fo;
my $t_first = @keep ? $keep[0][ $ix{t_syn} ]  : '';
my $t_last  = @keep ? $keep[-1][ $ix{t_syn} ] : '';

my $af_in = File::Spec->catfile($tmp, 'a_full.tsv');
open(my $ai, '<', $af_in) or die "Missing $af_in\n";
binmode $ai;
<$ai>;
my %pos;
while (1) {
    my $p = tell $ai;
    my $l = <$ai>;
    last unless defined $l;
    my ($cid) = $l =~ /^(\d+)\t/;
    $pos{ $sid_of{$cid} } = $p if defined $cid && exists $sid_of{$cid};
}
my $af_out = File::Spec->catfile($out, "A_full_$c->{SAMPLE}.tsv");
open(my $ao, '>', $af_out) or die "$!\n";
binmode $ao;
print $ao "session_id\ta_len_total\ta_full_hex\n";
for my $s (1 .. $sid) {
    die "A_full row missing for session $s\n" unless defined $pos{$s};
    seek($ai, $pos{$s}, 0);
    my $l = <$ai>;
    $l =~ s/\r?\n$//;
    $l =~ s/^\d+\t//;
    print $ao "$s\t$l\n";
}
close $ai; close $ao;

my $config_copy = File::Spec->catfile($out, basename($c->{_CONFIG_PATH}));
unless (-e $config_copy && Cwd::abs_path($config_copy) eq Cwd::abs_path($c->{_CONFIG_PATH})) {
    copy($c->{_CONFIG_PATH}, $config_copy) or die "config copy: $!\n";
}
my $sum = File::Spec->catfile($out, 'summary.txt');
open(my $so, '>', $sum) or die "$!\n";
binmode $so;
print $so "# 01_extract summary - $c->{DATASET} / $c->{SAMPLE} - written " . now_str() . "\n";
for my $st (qw(s1 s2 s4)) {
    print $so "\n[$st]\n";
    printf $so "%s = %s\n", @$_ for read_counts(File::Spec->catfile($tmp, "counts_$st.txt"));
}
print $so "\n[s5 selection]\n";
printf $so "usable_candidates = %d\nselected = %d\nN = %d\nt_syn_first = %s\nt_syn_last = %s\n",
    $available, $selected, $N, $t_first, $t_last;
printf $so "label_count_%s = %d\n", $_, $lab{$_} for sort keys %lab;
printf $so "attack_cat_%s = %d\n", $_, $cat{$_} for sort { $cat{$b} <=> $cat{$a} } keys %cat;
print $so "\n[s5 checks]\n";
printf $so "%s = %d\n", $_, $chk{$_} for sort keys %chk;
close $so;

logmsg("wrote $csv");
logmsg("wrote $af_out");
logmsg("wrote $sum");
logmsg('s5_select.pl done');

}

package main;
use strict;
use warnings;
use JSON::PP;
use Getopt::Long qw(GetOptions);
use File::Spec;
use File::Basename qw(dirname);
use File::Path qw(make_path);
use File::Copy qw(copy);
use Cwd qw(abs_path);

sub first_time {
    my ($tool,$path)=@_;
    open(my $fh,'-|',$tool,'-r',$path,'-c','1','-T','fields','-e','frame.time_epoch') or die "$!\n";
    my $value=<$fh>; close($fh) or die "tshark failed for $path\n";
    die "Empty capture: $path\n" unless defined($value);
    return $value+0;
}

# Select an hourly capture prefix without modifying the source captures.
sub slice_window {
    my ($c,$work,$editcap)=@_;
    return unless $c->{WINDOW_HOUR};
    my $pdir=$c->{PCAP_DIR};
    my $start=first_time($c->{TSHARK},File::Spec->catfile($pdir,$c->{WINDOW_T0_PCAP}))
        + ($c->{WINDOW_HOUR}-1)*3600;
    my $end=$start+($c->{WINDOW_SLICE_S}//7200);
    my $src=File::Spec->catfile($pdir,$c->{WINDOW_SOURCE});
    die "WINDOW_SOURCE starts after the selected hour\n" if first_time($c->{TSHARK},$src)>$start;
    my @files=split /,\s*/,$c->{PCAP_FILES};
    if (@files>1 && first_time($c->{TSHARK},File::Spec->catfile($pdir,$files[1])) <= $start) {
        die "WINDOW_SOURCE precedes the file containing the selected hour\n";
    }
    my $dest=File::Spec->catfile($work,'window.pcapng');
    system($editcap,'-A',sprintf('%.6f',$start),'-B',sprintf('%.6f',$end),$src,$dest)==0
        or die "editcap failed\n";
    my $first=first_time($c->{TSHARK},$dest);
    die "Unexpected first timestamp in window\n" unless $first >= $start && $first <= $start+600;
    $files[0]=$dest;
    $c->{PCAP_FILES}=join(',',@files);
}

my ($dataset,$sample,$raw,$output,$tshark,$editcap,$stage,$conf,$help);
$tshark='tshark'; $editcap='editcap';
GetOptions('dataset=s'=>\$dataset,'sample=s'=>\$sample,'raw=s'=>\$raw,
    'output=s'=>\$output,'tshark=s'=>\$tshark,'editcap=s'=>\$editcap,
    'stage=s'=>\$stage,'config=s'=>\$conf,'help'=>\$help) or die "Invalid arguments\n";
if ($stage) {
    die "--config required\n" unless $conf;
    @ARGV=($conf);
    if ($stage eq 'packets') { Stage_packets::run() }
    elsif ($stage eq 'sessions') { Stage_sessions::run() }
    elsif ($stage eq 'labels') { Stage_labels::run() }
    elsif ($stage eq 'select') { Stage_select::run() }
    else { die "Unknown stage\n" }
    exit;
}
if ($help || !$dataset || !$sample || !$raw) {
    print "perl extract.pl --dataset unsw_nb15|cicids2017 --sample W1|W2|W3|W4 --raw RAW_DIRECTORY [--output DIRECTORY] [--tshark PATH] [--editcap PATH]\n";
    exit($help ? 0 : 2);
}
my $root=dirname(abs_path(__FILE__));
open(my $sf,'<',File::Spec->catfile($root,'settings.json')) or die "$!\n";
my $settings=decode_json(do {local $/; <$sf>}); close $sf;
die "Unknown dataset/sample\n" unless exists $settings->{samples}{$dataset}{$sample};
my $c=$settings->{samples}{$dataset}{$sample}{extraction};
$raw=abs_path($raw) or die "Raw capture directory missing\n";
$output//=File::Spec->catdir($root,'generated','extraction');
make_path($output); $output=abs_path($output);
my $work=File::Spec->catdir($output,$dataset,$sample);
die "Output already exists; choose a new --output directory\n" if -e $work;
make_path($work);
my $pcaprel=$c->{PCAP_DIR}; $pcaprel =~ s{^raw_data/?}{};
$c->{PCAP_DIR}=File::Spec->catdir($raw,$pcaprel);
$c->{LABEL_FILES}=join(',',map {s{^raw_data/?}{}; File::Spec->catfile($raw,$_)} split /,\s*/,$c->{LABEL_FILES});
$c->{TSHARK}=$tshark; $c->{OUT_DIR}=$work;
$c->{TMP_DIR}=File::Spec->catdir($work,'intermediate'); $c->{LOG_DIR}=$work;
slice_window($c,$work,$editcap);
my $cfg=File::Spec->catfile($work,'run.conf');
open(my $cf,'>',$cfg) or die "$!\n";
print $cf "$_ = $c->{$_}\n" for sort keys %$c; close $cf;
for my $st (qw(packets sessions labels select)) {
    system($^X,abs_path(__FILE__),'--stage',$st,'--config',$cfg)==0 or die "Extraction stage $st failed\n";
}
my $csv=File::Spec->catfile($work,"sessions_$c->{SAMPLE}.csv");
print "Extracted sessions: $csv\n";
