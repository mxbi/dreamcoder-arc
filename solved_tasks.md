## ARC-Easy

`arcbin/arc_mikel2.py -c 76 -t 3600 -R 2400 -i 1`

```
2107 Test set evaluation
2108 HIT @ 1 for 00d62c1b with (fillobj c4 $0)
2109 HIT @ 1 for 06df4c85 with (overlay (ic_connectY (ic_erasecol (topcol $0) $0)) $0)
2110 Exception pickunique: no unique grids for 0b148d64 with (ic_pickunique (split8 (ic_erasecol c1 $0)))
2111 FAIL: Evaluated 5 solns for task 0b148d64, no successes.
2112 HIT @ 1 for 1190e5a7 with (ic_compress2 (ic_compress3 (ic_erasecol (rarestcol $0) $0)))
2113 HIT @ 1 for 1cf80156 with (ic_compress2 $0)
2114 HIT @ 1 for 1e0a9b12 with (gravity_down $0)
2115 HIT @ 1 for 1f85a75f with (ic_pickunique (split8 $0))
2116 HIT @ 1 for 2013d3e2 with (left_half (top_half (ic_compress2 $0)))
2117 HIT @ 1 for 22168020 with (overlay $0 (ic_connectX $0))
2118 HIT @ 1 for 22eb0ac0 with (overlay $0 (ic_connectX $0))
2119 HIT @ 1 for 23b5c85d with (ic_compress2 (ic_filtercol (rarestcol $0) $0))
2120 HIT @ 1 for 253bf280 with (overlay $0 (ic_connectY (setcol c3 $0)))
2121 HIT @ 1 for 28bf18c6 with (ic_compress2 (repeatX $0))
2122 HIT @ 1 for 2dee498d with (ic_compress2 (ic_embed $0 (rot270 $0)))
2123 HIT @ 1 for 39a8645d with (pickcommon (split8 $0))
2124 HIT @ 1 for 3aa6fb7a with (overlay $0 (mapSplit8 (lambda (ic_makeborder $0)) $0))
2125 HIT @ 1 for 3af2c5a8 with (mirrorX (mirrorY $0))
2126 HIT @ 1 for 3c9b0459 with (rot180 $0)
2127 HIT @ 1 for 40853293 with (overlay $0 (ic_connectY $0))
2128 HIT @ 1 for 4258a5f9 with (fillobj c5 (ic_makeborder $0))
2129 HIT @ 1 for 4347f46a with (logical_and $0 (ic_makeborder (ic_invert $0)))
2130 HIT @ 1 for 445eab21 with (countToXY (countColours $0) (topcol $0))
2131 HIT @ 1 for 496994bd with (mirrorY (top_half $0))
2132 HIT @ 1 for 4c4377d9 with (mirrorY (flipx $0))
2133 HIT @ 1 for 5582e5ca with (colourHull (topcol $0) $0)
2134 HIT @ 1 for 60b61512 with (overlay $0 (mapSplit8 (lambda (set_bg c7 $0)) $0))
2135 HIT @ 1 for 6150a2bd with (rot180 $0)
2136 HIT @ 1 for 62c24649 with (mirrorX (mirrorY $0))
2137 HIT @ 1 for 67a3c6ac with (flipy $0)
2138 HIT @ 1 for 67e8384a with (mirrorX (mirrorY $0))
2139 HIT @ 1 for 68b16354 with (flipx $0)
2140 HIT @ 1 for 6d0aefbc with (mirrorX $0)
2141 HIT @ 1 for 6d75e8bb with (overlay $0 (set_bg c2 (ic_composegrowing (ic_splitall $0))))
2142 HIT @ 1 for 6fa7a44f with (mirrorY $0)
2143 HIT @ 1 for 72ca375d with (pickcommon (ic_splitall (mirrorX $0)))
2144 HIT @ 1 for 7468f01a with (flipy (ic_compress2 $0))
2145 HIT @ 1 for 746b3537 with (ic_compress3 $0)
2146 HIT @ 1 for 74dd1130 with (swapxy $0)
2147 HIT @ 1 for 7b6016b9 with (set_bg c3 (fillobj c2 $0))
2148 HIT @ 1 for 8be77c9e with (mirrorY $0)
2149 HIT @ 1 for 90c28cc7 with (ic_compress2 (ic_compress3 $0))
2150 HIT @ 1 for 9565186b with (set_bg c5 (ic_filtercol (topcol $0) $0)) 
2151 HIT @ 1 for 9dfd6313 with (swapxy $0)
2152 HIT @ 1 for a416b8f3 with (repeatX $0)
2153 HIT @ 1 for a5313dff with (fillobj c1 $0)
2154 HIT @ 1 for a699fb00 with (overlay $0 (ic_connectX (setcol c2 $0)))
2155 HIT @ 1 for a740d043 with (ic_composegrowing (ic_splitall (ic_erasecol c1 $0)))
2156 HIT @ 1 for a87f7484 with (ic_compress2 (ic_filtercol (topcol $0) $0))
2157 HIT @ 1 for aabf363d with (ic_connectX (get_bg (rarestcol $0) $0))
2158 HIT @ 1 for b1948b0a with (set_bg c2 (ic_erasecol c6 $0))
2159 HIT @ 3 for b8825c91 with (overlay $0 (ic_erasecol c4 (rot180 $0)))
2160 HIT @ 1 for b9b7f026 with (ic_compress2 (ic_compress3 (ic_connectX $0)))
2161 FAIL: Evaluated 5 solns for task be94b721, no successes.
2162 HIT @ 1 for c8f0f002 with (set_bg c5 (ic_erasecol c7 $0))
2163 FAIL: Evaluated 5 solns for task c909285e, no successes.
2164 HIT @ 1 for c9e6f938 with (mirrorX $0)
2165 HIT @ 1 for ce22a75a with (ic_fill (ic_makeborder $0))
2166 HIT @ 1 for d037b0a7 with (ic_embed (overlay $0 (ic_connectY (mirrorY $0))) $0)
2167 HIT @ 1 for d10ecb37 with (ic_embed $0 (gravity_down (ic_center $0)))
2168 HIT @ 2 for d23f8c26 with (ic_embed (pickmax_cols (ic_splitcolumns (right_half $0))) $0)
2169 HIT @ 1 for d5d6de2d with (ic_connectX (setcol c3 $0))
2170 HIT @ 1 for d631b094 with (countToY (countPixels $0) (rarestcol $0))
2171 HIT @ 1 for d9fac9be with (pickcommon (split8 $0))
2172 HIT @ 1 for dae9d2b5 with (setcol c6 (overlay (left_half $0) (right_half $0)))
2173 HIT @ 1 for dbc1a6ce with (overlay $0 (ic_connectY (setcol c8 $0)))
2174 HIT @ 1 for dc1df850 with (overlay $0 (ic_makeborder (pickmax_cols (ic_splitcols $0))))
2175 FAIL: Evaluated 2 solns for task de1cd16c, no successes.
2176 HIT @ 1 for ded97339 with (overlay $0 (ic_connectY $0))
2177 HIT @ 1 for e3497940 with (left_half (overlay $0 (flipy $0)))
2178 FAIL: Evaluated 1 solns for task e50d258f, no successes.
2179 HIT @ 1 for eb5a1d5d with (ic_compress3 $0)
2180 HIT @ 1 for ed36ccf7 with (rot90 $0)
2181 HIT @ 1 for f25ffba3 with (ic_compress2 (mirrorY (flipx $0)))
2182 HIT @ 1 for f76d97a5 with (ic_invert (ic_erasecol c5 $0))
2183 HIT @ 1 for fcb5c309 with (setcol (rarestcol $0) (pickmax_cols (ic_splitall $0)))
2184 Test summary: 68 (17.0%) acc@1, 70 (17.5%) acc@3 
```

## ARC-Hard

`arcbin/arc_mikel2.py -c 76 -t 3600 -R 2400 -i 1 --evalset`

```
1738 Test set evaluation
1739 HIT @ 1 for 070dd51e with (overlay $0 (ic_connectY $0))
1740 HIT @ 1 for 0c786b71 with (mirrorX (mirrorY (rot180 $0)))
1741 HIT @ 1 for 19bb5feb with (ic_compress2 (ic_compress3 (ic_erasecol c8 $0)))
1742 HIT @ 1 for 1a2e2828 with (ic_compress3 (ic_pickunique (ic_splitall $0)))
1743 HIT @ 1 for 3194b014 with (countToXY (countColours (pickmax_interior_count (split8 $0))) (topcol (pickmax_interior_count (ic_splitall $0))))
1744 HIT @ 1 for 59341089 with (mirrorX (mirrorX (flipy $0)))
1745 HIT @ 1 for 5ffb2104 with (gravity_right $0)
1746 FAIL: Evaluated 5 solns for task 642d658d, no successes.
1747 HIT @ 1 for 68b67ca3 with (ic_compress2 $0)
1748 HIT @ 1 for 73182012 with (left_half (top_half (ic_compress2 $0)))
1749 FAIL: Evaluated 5 solns for task 73ccf9c2, no successes.
1750 HIT @ 1 for 833dafe3 with (mirrorX (mirrorY (rot180 $0)))
1751 HIT @ 1 for 84db8fc4 with (set_bg c2 (fillobj c5 $0))
1752 FAIL: Evaluated 4 solns for task 9ddd00f0, no successes.
1753 HIT @ 1 for aa18de87 with (overlay $0 (ic_connectX (setcol c2 $0)))
1754 HIT @ 1 for be03b35f with (left_half (rot180 (top_half $0)))
1755 HIT @ 1 for bf32578f with (ic_connectX (mirrorX (left_half $0)))
1756 FAIL: Evaluated 5 solns for task bf699163, no successes.
1757 HIT @ 1 for cd3c21df with (ic_pickunique (ic_splitall $0))
1758 HIT @ 1 for ce8d95cc with (ic_compress3 $0)
1759 Exception Grid size (19, 36) too large for d56f2372 with (pickcommon (split8 (mirrorX (ic_compress2 $0))))
1760 Exception Grid size (19, 36) too large for d56f2372 with (pickcommon (split8 (mirrorX (ic_composegrowing (ic_splitall $0)))))
1761 Exception Grid size (19, 36) too large for d56f2372 with (pickcommon (split8 (mirrorX (ic_composegrowing (split8 $0)))))
1762 Exception Grid size (19, 36) too large for d56f2372 with (pickcommon (split8 (mirrorX (mapSplit8 (lambda $0) $0))))
1763 FAIL: Evaluated 5 solns for task d56f2372, no successes.
1764 HIT @ 1 for e1baa8a4 with (ic_compress3 $0)
1765 HIT @ 1 for f0df5ff0 with (overlay (ic_makeborder (ic_filtercol c1 $0)) $0)
1766 Test summary: 18 (4.5%) acc@1, 18 (4.5%) acc@3 
```