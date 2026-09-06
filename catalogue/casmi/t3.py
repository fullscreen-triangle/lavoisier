import time, json, identify as I
d = json.load(open('casmi_spectra.json'))
ch = [x for x in d if x['id'] == 3][0]
pol = ch['scans'][0]['polarity']
frags, _ = I.merge_ladder(ch['scans'])
t = time.time(); c = I.d1_mass(ch['mz'], pol)
print('d1 %.1fs cands=%d' % (time.time() - t, len(c)), flush=True)
for cand in c[:6]:
    p = dict(cand['formula'])
    if cand['nmer'] > 1:
        p = {e: n * cand['nmer'] for e, n in p.items()}
    sz = 1
    for e, n in p.items():
        sz *= (n + 1)
    print('  %-14s space=%d' % (cand['fstr'], sz), flush=True)
    t = time.time(); idx = I.SubIndex(p)
    print('     SubIndex %.2fs entries=%d' % (time.time() - t, len(idx.masses)), flush=True)
