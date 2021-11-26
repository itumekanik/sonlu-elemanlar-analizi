class Plot:
    def __init__(self, elements, func, triangle_map, line_map, node_cls, coords0, coords1, conn_name='nodes', mesh=True,
                 name="", unit="", lookat=[0,0,1]):
        XYZ = [[getattr(n, coords0[0]), getattr(n, coords0[1]), getattr(n, coords0[2])] for n in node_cls]
        xyz = [[getattr(n, coords1[0]), getattr(n, coords1[1]), getattr(n, coords1[2])] for n in node_cls]
        from operator import itemgetter

        NC = 25# Number of maximum contour points in positive or negative direction

        def interpolate(val, y0, x0, y1, x1):
            return (val - x0) * (y1 - y0) / (x1 - x0) + y0

        def base(val):
            if val <= -0.75: return 0
            if val <= -0.25: return interpolate(val, 0.0, -0.75, 1.0, -0.25)
            if val <= 0.25: return 1.0
            if val <= 0.75: return interpolate(val, 1.0, 0.25, 0.0, 0.75)
            return 0.0

        def red(gray):
            return base(gray - 0.5)

        def green(gray):
            return base(gray)

        def blue(gray):
            return base(gray + 0.5)

        import numpy as np
        cmap = [[red(gval), green(gval), blue(gval)] for gval in np.linspace(1, -1, 2 * NC - 1)]  # *******

        #print("aaaaaaaaaaaa", [red(1-0.4), green(1-0.4), blue(1-0.4)])
        #print("aaaaaaaaaaaa", [red(1), green(1), blue(1)])

        cells = {}
        lines = {}
        for s in elements:
            vals = getattr(s, func)()  # taking nodal values
            #indexes = list(s.nodes[:]._index)  # calculating global node indexes
            indexes = [node._index for node in getattr(s, conn_name)]
            # indexes = list(getattr(s, conn_name)[:]._index)
            # indexes = [n._index for n in s.conn]
            for map in triangle_map:
                c = tuple(i for i in itemgetter(*map)(indexes))  # calculate triangle's global indexes
                key = tuple(sorted(c))
                if key in cells.keys():
                    del cells[key]
                else:
                    cells[key] = c, s, map, itemgetter(*map)(vals)  # degerler son eleman olmali
            if not mesh:
                #print(indexes)
                for lmap in line_map:
                    cl = tuple(i for i in itemgetter(*lmap)(indexes))
                    key = tuple(sorted(cl))
                    if key in lines.keys():
                        del lines[key]
                    else:
                        lines[key] = cl

        # Mesh Lines
        if mesh:
            for c in cells.values():
                # indexes = list(c[1].nodes[:]._index)  # calculating global node indexes
                # indexes = list(getattr(c[1], conn_name)[:]._index)  # calculating global node indexes
                # indexes = [n._index for n in c[1].conn]
                indexes = [node._index for node in getattr(c[1], conn_name)]
                for lmap in line_map:
                    if len(set(list(lmap) + list(c[2]))) == len(c[2]):  # Eğer mapler uyusuyor ise:
                        cl = tuple(i for i in itemgetter(*lmap)(indexes))
                        key = tuple(sorted(cl))
                        if key in lines.keys():
                            pass
                            # del lines[key]
                        else:
                            lines[key] = cl

        _max = max(max(c[-1]) for c in cells.values())
        _min = min(min(c[-1]) for c in cells.values())

        if _max == _min:
            if _max == 0: _max = 1
            pass  # Set color identifier funnction eitler positive/negative or zero...
        _amax = max(abs(_min), abs(_max))
        _max = _amax
        _min = -_amax
        # print(_min, _max)

        dv = (_max - _min) / (2 * NC - 1)
        cps = [(i + 0.5) * dv for i in range(NC)]  # contour points for positive
        cpsn = [(i + 0.5) * dv / _amax for i in range(NC)]  # normalized contour points for positive
        #print(cps)
        #print(cpsn)

        def vcolor(cpsn, amax):
            def get(val):
                factor = 1
                if val < 0: factor = -1
                try:
                    return [factor * (i + 1) for i, c in enumerate(cpsn) if c < abs(val) / amax][-1]
                except:
                    return 0

            return get

        _vc = vcolor(cpsn, _amax)

        def cpoints(cpsn, amax):
            N = len(cpsn)
            full_cpsn = list(reversed([-c for c in cpsn])) + cpsn

            def get(val):
                factor = 1
                if val < 0: factor = -1
                return set([factor * c for c in cpsn if c < abs(val) / amax])

            return get

        def normalized(amax):
            return lambda val: val / amax

        _cpsn = cpoints(cpsn, _amax)
        _n = normalized(_amax)

        #---------------------------------------

        cvals = [(i + 0.5) * dv for i in range(NC)]
        cvals_all = list(reversed([-c for c in cvals])) + cvals

        def get_coord(c1, c2, v1, v2, val):
            f = (val - v1) / (v2 - v1)
            return [c1[i] + (c2[i] - c1[i]) * f for i in range(3)]

        def get_contour_vals_between(v1, v2):
            return [v for v in cvals_all if v1 < v < v2]

        def get_triangles(indexes, vals):
            sorted_items = sorted([(i, v) for i, v in zip(indexes, vals)], key=lambda t: t[1])
            c0, c1, c2 = (xyz[i] for i, v in sorted_items)
            v0, v1, v2 = (v for i, v in sorted_items)
            v3 = v1
            c3 = get_coord(c0, c2, v0, v2, v3)

            vals1 = get_contour_vals_between(v0, v1)
            vals2 = get_contour_vals_between(v1, v2)

            new_xyz = [c0, c1, c2, c3]
            new_trig = []
            new_vals = [v0, v1, v2, v3]

            for v in vals1:
                new_xyz.append(get_coord(c0, c3, v0, v3, v))
                new_xyz.append(get_coord(c0, c1, v0, v1, v))
                new_vals.append(v)
                new_vals.append(v)

            len1 = len(vals1)

            if len1 == 0:
                new_trig.append([0, 3, 1])

            if len1 == 1:
                new_trig.append([0, 4, 5])
                new_trig.append([4, 5, 3])
                new_trig.append([3, 5, 1])

            if len1 > 1:
                new_trig.append([0, 4, 5])
                for i in range(len(vals1) - 1):
                    new_trig.append([4 + 2 * i, 5 + 2 * i, 6 + 2 * i])
                    new_trig.append([6 + 2 * i, 7 + 2 * i, 5 + 2 * i])
                last = 4 + 2 * (len(vals1) - 1)
                new_trig.append([last, last + 1, 3])
                new_trig.append([3, last + 1, 1])

            for v in reversed(vals2):
                new_xyz.append(get_coord(c2, c3, v2, v3, v))
                new_xyz.append(get_coord(c2, c1, v2, v1, v))
                new_vals.append(v)
                new_vals.append(v)

            last = 4 + 2 * len1

            len2 = len(vals2)

            if len2 == 0:
                new_trig.append([3, 2, 1])

            if len2 == 1:
                new_trig.append([2, last, last + 1])
                new_trig.append([3, last, 1])
                new_trig.append([last, last + 1, 1])

            if len2 > 1:
                new_trig.append([2, last, last + 1])
                for i in range(len(vals2) - 1):
                    new_trig.append([last + 2 * i, last + 1 + 2 * i, last + 2 + 2 * i])
                    new_trig.append([last + 2 + 2 * i, last + 3 + 2 * i, last + 1 + 2 * i])
                    pass
                last = 4 + 2 * len1 + 2 * (len2 - 1)

                new_trig.append([last, last + 1, 3])
                new_trig.append([3, 1, last + 1])

            return new_xyz, new_trig, new_vals
        #---------------------------------------



        # BURADA TEEEEEEEEEEEEEEEEEESSSTT

        ccells = []  # calculated triangle veritices global indexeses
        clr = []  # calculated color indexses
        next_xyz_index = len(xyz)
        for key, item in cells.items():
            cc = [_vc(v) for v in item[-1]]  # color codes
            if len([c for c in cc if c != cc[0]]) == 0:  # Tum renk kodlari ayni ise
                ccells.append(list(item[0]))
                clr.append(cc[0] + (NC - 1))
            else:  # add new triangles
                #ccells.append(list(item[0]))
                #clr.append(_vc(sum(item[-1]) / 3) + (NC - 1))
                #continue
                new_xyz, new_trigs, new_vals = get_triangles(item[0], item[-1])

                colors = np.asarray([_vc(np.mean(np.asarray(new_vals)[trig])) for trig in new_trigs])

                for e in new_xyz:
                    xyz.append(e)
                for t, c in zip(new_trigs, colors):
                    ccells.append([t[0] + next_xyz_index,
                                   t[1] + next_xyz_index,
                                   t[2] + next_xyz_index])

                    clr.append(c + (NC -1))
                next_xyz_index += len(new_xyz)
        # BURADA TEEEEEEEEEEEEEEEEEESSSTT

        ccells = []  # calculated triangle veritices global indexeses
        clr = []  # calculated color indexses
        next_xyz_index = len(xyz)
        for key, item in cells.items():
            cc = [_vc(v) for v in item[-1]]  # color codes
            if len([c for c in cc if c != cc[0]]) == 0:  # Tum renk kodlari ayni ise
                ccells.append(list(item[0]))
                clr.append(cc[0] + (NC - 1))
            else:  # add new triangles
                #ccells.append(list(item[0]))
                #clr.append(_vc(sum(item[-1]) / 3) + (NC - 1))
                #continue
                new_xyz, new_trigs, new_vals = get_triangles(item[0], item[-1])

                colors = np.asarray([_vc(np.mean(np.asarray(new_vals)[trig])) for trig in new_trigs])

                for e in new_xyz:
                    xyz.append(e)
                for t, c in zip(new_trigs, colors):
                    ccells.append([t[0] + next_xyz_index,
                                   t[1] + next_xyz_index,
                                   t[2] + next_xyz_index])

                    clr.append(c + (NC -1))
                next_xyz_index += len(new_xyz)




        outer_lines = [list(item) for item in lines.values()]

        outer_lines = "var outer=" + str(outer_lines) + ";\n"
        cmap = "var cmap=" + str(cmap) + ";\n"
        ccells = "var cell=" + str(ccells) + ";\n"
        clr = "var clr=" + str(clr) + ";\n"
        xyz = "var xyz=" + str(xyz) + ";\n"
        XYZ = "var XYZ=" + str(XYZ) + ";\n"

        f = open(r'.\drawing\DRAW\data\data.js', 'w')

        f.write("var function_name = '"+ name + "'\n")
        f.write("var function_unit = '"+ unit + "'\n")
        f.write("var function_max = " + str(_amax) + "\n")
        f.write("var lookat = " + str(lookat) + "\n")

        f.write(cmap)
        f.write(xyz)
        f.write(XYZ)
        f.write(ccells)
        f.write(clr)
        f.write((outer_lines))
        f.close()

        #import time
        #time.sleep(2)

        import os
        os.startfile(r'.\drawing\DRAW\index.html')
