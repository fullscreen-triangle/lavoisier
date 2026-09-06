import time, json, identify as I
d = json.load(open('casmi_spectra.json'))
ch = d[0]
print('first challenge id', ch['id'], 'mz', ch['mz'], flush=True)
pol = ch['scans'][0]['polarity']
frags, _ = I.merge_ladder(ch['scans'])
t = time.time(); allc = I.d1_mass(ch['mz'], pol); print('d1 %.1fs n=%d' % (time.time()-t, len(allc)), flush=True)
t = time.time(); c, n = I.prefilter(allc); print('prefilter %.1fs kept=%d' % (time.time()-t, len(c)), flush=True)
t = time.time(); used = I.d2_contact(c, frags, ch['mz'], pol); print('d2 %.1fs' % (time.time()-t), flush=True)
