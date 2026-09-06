import time, json, identify as I
d = json.load(open('casmi_spectra.json'))
ch = [x for x in d if x['id'] == 42][0]
pol = ch['scans'][0]['polarity']
frags, ces = I.merge_ladder(ch['scans'])
print('frags', len(frags), flush=True)
t = time.time(); c = I.d1_mass(ch['mz'], pol)
print('d1 %.1fs cands=%d distinct=%d' % (time.time() - t, len(c), len({x["fstr"] for x in c})), flush=True)
use = [f for f in frags if abs(f["mz"] - ch['mz']) > 0.02][:25]
print('use', len(use), flush=True)
t = time.time(); idx = I.SubIndex(c[0]['formula']); print('SubIndex %.2fs size=%d for %s' % (time.time() - t, len(idx.masses), c[0]['fstr']), flush=True)
t = time.time()
for fr in use[:5]:
    idx.match(fr['mz'] + I.E, fr['mz'] * 12e-6)
print('5 matches %.2fs' % (time.time() - t), flush=True)
