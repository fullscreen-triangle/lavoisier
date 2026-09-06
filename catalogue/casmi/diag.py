import json, identify as I
d = json.load(open('casmi_spectra.json'))
ch = [x for x in d if x['id'] == 42][0]
pol = ch['scans'][0]['polarity']
frags, _ = I.merge_ladder(ch['scans'])
allc = I.d1_mass(ch['mz'], pol)
print('all mass candidates for 199.0965 (+):')
for c in sorted(allc, key=lambda c: abs(c['ppm'])):
    print('  %-12s %-14s ppm=%+.2f rdbe=%.1f' % (c['adduct'], c['fstr'], c['ppm'], I.rdbe(c['formula'])))
print()
print('top fragments:')
for f in frags[:12]:
    print('  %.4f rel=%.3f n_ce=%d' % (f['mz'], f['rel'], f['n_ce']))
