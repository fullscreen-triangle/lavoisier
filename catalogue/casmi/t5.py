import time, bisect, json, identify as I
d = json.load(open('casmi_spectra.json'))
ch = d[0]
pol = ch['scans'][0]['polarity']
frags, _ = I.merge_ladder(ch['scans'])
allc = I.d1_mass(ch['mz'], pol); c, n = I.prefilter(allc)
use = [f for f in frags if abs(f["mz"] - ch['mz']) > 0.02][:25]
tot_e = 0
for cand in c[:8]:
    p = dict(cand['formula'])
    t = time.time(); idx = I.SubIndex(p); tb = time.time()-t
    t = time.time()
    win = 0
    for fr in use:
        fm = fr['mz'] - I.E
        tol = fm * 12e-6
        lo = bisect.bisect_left(idx.masses, fm - tol); hi = bisect.bisect_right(idx.masses, fm + tol)
        win += hi - lo
    tm = time.time()-t
    print('%-14s entries=%7d build=%.2fs  window_total=%6d match=%.2fs' % (cand['fstr'], len(idx.masses), tb, win, tm), flush=True)
