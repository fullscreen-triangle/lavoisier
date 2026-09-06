import time, json, identify as I
d = json.load(open('casmi_spectra.json'))
for cid in (42, 3):
    ch = [x for x in d if x['id'] == cid][0]
    pol = ch['scans'][0]['polarity']
    frags, ces = I.merge_ladder(ch['scans'])
    t = time.time(); c = I.d1_mass(ch['mz'], pol); t1 = time.time() - t
    t = time.time(); used = I.d2_contact(c, frags, ch['mz'], pol); t2 = time.time() - t
    v, b, r = I.license_answer(c)
    print('#%d mz=%.4f %s nfrag=%d cands=%d distinct=%d d1=%.1fs d2=%.1fs' % (
        cid, ch['mz'], pol, len(used), len(c), len({x["fstr"] for x in c}), t1, t2))
    print('   ->', v, b and b['adduct'], b and b['fstr'], 'contact=%.3f' % (b['contact'] if b else 0))
