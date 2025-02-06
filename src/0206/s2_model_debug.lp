\ Model profit_maximization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  profits_vars[0] + profits_vars[1] + profits_vars[2] + profits_vars[3]
   + profits_vars[4] + profits_vars[5] + profits_vars[6] + profits_vars[7]
   + profits_vars[8] + profits_vars[9] + profits_vars[10]
   + profits_vars[11] + profits_vars[12] + profits_vars[13]
   + profits_vars[14] + profits_vars[15] + profits_vars[16]
   + profits_vars[17] + profits_vars[18] + profits_vars[19]
   + profits_vars[20] + profits_vars[21] + profits_vars[22]
   + profits_vars[23] + profits_vars[24]
Subject To
 Fraction_lower_bound_0: Fraction[0] >= 0
 Fraction_upper_bound_0: Fraction[0] <= 1
 R2: 0.1485442135232429 alphas[0] - 0.3851892822364052 alphas[1]
   - 0.2394304242396365 alphas[2] + 1.04415860753828 alphas[3] - alphas[4]
   + f_var[0] = 0
 R3: - 2053.400566507637 Fraction[0] + Q0_var[0] = 0
 R4: Q0_var[0] + Lost_0_expr_0 = 1626.043314427274
 R5: Q0_var[0] + Q_hat_adjusted_0 = 2064.951581619729
 R6: Sold_0[0] - Q0_var[0] - Q1_var[0] + total_stock_second_period_0 = 0
 R7: total_stock_second_period_0 + Lost_1_expr_0 = 438.9082683981151
 R8: Sold_1[0] + Left_1[0] - total_stock_second_period_0 = 0
 R9: - 500 Sold_0[0] - 500 Sold_1[0] + 500 Lost_0[0] + 500 Lost_1[0]
   + 500 Left_1[0] + profits_vars[0] = 0
 Fraction_lower_bound_1: Fraction[1] >= 0
 Fraction_upper_bound_1: Fraction[1] <= 1
 R12: - 0.4405511694217167 alphas[0] + 0.0486297156342791 alphas[1]
   + 1.57241865352909 alphas[2] - 1.211868098505336 alphas[3] - alphas[4]
   + f_var[1] = 0
 R13: - 2053.400566507637 Fraction[1] + Q0_var[1] = 0
 R14: Q0_var[1] + Lost_0_expr_1 = 739.4991742575022
 R15: Q0_var[1] + Q_hat_adjusted_1 = 974.5031660793892
 R16: Sold_0[1] - Q0_var[1] - Q1_var[1] + total_stock_second_period_1 = 0
 R17: total_stock_second_period_1 + Lost_1_expr_1 = 235.0039865960826
 R18: Sold_1[1] + Left_1[1] - total_stock_second_period_1 = 0
 R19: - 500 Sold_0[1] - 500 Sold_1[1] + 500 Lost_0[1] + 500 Lost_1[1]
   + 500 Left_1[1] + profits_vars[1] = 0
 Fraction_lower_bound_2: Fraction[2] >= 0
 Fraction_upper_bound_2: Fraction[2] <= 1
 R22: - 0.0424788118340407 alphas[0] - 1.763758393415958 alphas[1]
   - 0.4943523487684089 alphas[2] + 1.003209748914306 alphas[3] - alphas[4]
   + f_var[2] = 0
 R23: - 2053.400566507637 Fraction[2] + Q0_var[2] = 0
 R24: Q0_var[2] + Lost_0_expr_2 = 1598.520274925257
 R25: Q0_var[2] + Q_hat_adjusted_2 = 2066.117285596097
 R26: Sold_0[2] - Q0_var[2] - Q1_var[2] + total_stock_second_period_2 = 0
 R27: total_stock_second_period_2 + Lost_1_expr_2 = 467.5970110612975
 R28: Sold_1[2] + Left_1[2] - total_stock_second_period_2 = 0
 R29: - 500 Sold_0[2] - 500 Sold_1[2] + 500 Lost_0[2] + 500 Lost_1[2]
   + 500 Left_1[2] + profits_vars[2] = 0
 Fraction_lower_bound_3: Fraction[3] >= 0
 Fraction_upper_bound_3: Fraction[3] <= 1
 R32: 0.1624604984913878 alphas[0] + 1.158127845551552 alphas[1]
   - 1.502864317011586 alphas[2] - 0.4190277033472191 alphas[3] - alphas[4]
   + f_var[3] = 0
 R33: - 2053.400566507637 Fraction[3] + Q0_var[3] = 0
 R34: Q0_var[3] + Lost_0_expr_3 = 2398.802338453156
 R35: Q0_var[3] + Q_hat_adjusted_3 = 2979.896608353331
 R36: Sold_0[3] - Q0_var[3] - Q1_var[3] + total_stock_second_period_3 = 0
 R37: total_stock_second_period_3 + Lost_1_expr_3 = 581.0942725587191
 R38: Sold_1[3] + Left_1[3] - total_stock_second_period_3 = 0
 R39: - 500 Sold_0[3] - 500 Sold_1[3] + 500 Lost_0[3] + 500 Lost_1[3]
   + 500 Left_1[3] + profits_vars[3] = 0
 Fraction_lower_bound_4: Fraction[4] >= 0
 Fraction_upper_bound_4: Fraction[4] <= 1
 R42: 0.5916999328977518 alphas[0] + 0.8059439709717974 alphas[1]
   + 1.666851549198021 alphas[2] + 1.12460937732089 alphas[3] - alphas[4]
   + f_var[4] = 0
 R43: - 2053.400566507637 Fraction[4] + Q0_var[4] = 0
 R44: Q0_var[4] + Lost_0_expr_4 = 1363.144551235566
 R45: Q0_var[4] + Q_hat_adjusted_4 = 1587.52112455324
 R46: Sold_0[4] - Q0_var[4] - Q1_var[4] + total_stock_second_period_4 = 0
 R47: total_stock_second_period_4 + Lost_1_expr_4 = 224.376571759979
 R48: Sold_1[4] + Left_1[4] - total_stock_second_period_4 = 0
 R49: - 500 Sold_0[4] - 500 Sold_1[4] + 500 Lost_0[4] + 500 Lost_1[4]
   + 500 Left_1[4] + profits_vars[4] = 0
 Fraction_lower_bound_5: Fraction[5] >= 0
 Fraction_upper_bound_5: Fraction[5] <= 1
 R52: - 0.1951939812905471 alphas[0] + 0.9627548780597103 alphas[1]
   + 0.2122303512187826 alphas[2] - 1.148715514237678 alphas[3] - alphas[4]
   + f_var[5] = 0
 R53: - 2053.400566507637 Fraction[5] + Q0_var[5] = 0
 R54: Q0_var[5] + Lost_0_expr_5 = 1567.672756172823
 R55: Q0_var[5] + Q_hat_adjusted_5 = 1955.751423474789
 R56: Sold_0[5] - Q0_var[5] - Q1_var[5] + total_stock_second_period_5 = 0
 R57: total_stock_second_period_5 + Lost_1_expr_5 = 388.0786672089184
 R58: Sold_1[5] + Left_1[5] - total_stock_second_period_5 = 0
 R59: - 500 Sold_0[5] - 500 Sold_1[5] + 500 Lost_0[5] + 500 Lost_1[5]
   + 500 Left_1[5] + profits_vars[5] = 0
 Fraction_lower_bound_6: Fraction[6] >= 0
 Fraction_upper_bound_6: Fraction[6] <= 1
 R62: 0.5423687413596802 alphas[0] - 0.6585168211974903 alphas[1]
   + 0.1416568156698964 alphas[2] - 1.020986388394283 alphas[3] - alphas[4]
   + f_var[6] = 0
 R63: - 2053.400566507637 Fraction[6] + Q0_var[6] = 0
 R64: Q0_var[6] + Lost_0_expr_6 = 1672.528001503375
 R65: Q0_var[6] + Q_hat_adjusted_6 = 2068.548967509328
 R66: Sold_0[6] - Q0_var[6] - Q1_var[6] + total_stock_second_period_6 = 0
 R67: total_stock_second_period_6 + Lost_1_expr_6 = 396.0209656359195
 R68: Sold_1[6] + Left_1[6] - total_stock_second_period_6 = 0
 R69: - 500 Sold_0[6] - 500 Sold_1[6] + 500 Lost_0[6] + 500 Lost_1[6]
   + 500 Left_1[6] + profits_vars[6] = 0
 Fraction_lower_bound_7: Fraction[7] >= 0
 Fraction_upper_bound_7: Fraction[7] <= 1
 R72: - 1.065789723894727 alphas[0] + 0.6595252882541187 alphas[1]
   + 0.2295817456905412 alphas[2] - 0.2731871634863502 alphas[3]
   - alphas[4] + f_var[7] = 0
 R73: - 2053.400566507637 Fraction[7] + Q0_var[7] = 0
 R74: Q0_var[7] + Lost_0_expr_7 = 1279.491825681141
 R75: Q0_var[7] + Q_hat_adjusted_7 = 1665.617778576442
 R76: Sold_0[7] - Q0_var[7] - Q1_var[7] + total_stock_second_period_7 = 0
 R77: total_stock_second_period_7 + Lost_1_expr_7 = 386.1259528954367
 R78: Sold_1[7] + Left_1[7] - total_stock_second_period_7 = 0
 R79: - 500 Sold_0[7] - 500 Sold_1[7] + 500 Lost_0[7] + 500 Lost_1[7]
   + 500 Left_1[7] + profits_vars[7] = 0
 Fraction_lower_bound_8: Fraction[8] >= 0
 Fraction_upper_bound_8: Fraction[8] <= 1
 R82: - 1.320333411356792 alphas[0] - 0.0427170655502229 alphas[1]
   + 1.780907278927179 alphas[2] + 0.2350935931801337 alphas[3] - alphas[4]
   + f_var[8] = 0
 R83: - 2053.400566507637 Fraction[8] + Q0_var[8] = 0
 R84: Q0_var[8] + Lost_0_expr_8 = 709.2630595513233
 R85: Q0_var[8] + Q_hat_adjusted_8 = 920.8038788696135
 R86: Sold_0[8] - Q0_var[8] - Q1_var[8] + total_stock_second_period_8 = 0
 R87: total_stock_second_period_8 + Lost_1_expr_8 = 211.5408163203091
 R88: Sold_1[8] + Left_1[8] - total_stock_second_period_8 = 0
 R89: - 500 Sold_0[8] - 500 Sold_1[8] + 500 Lost_0[8] + 500 Lost_1[8]
   + 500 Left_1[8] + profits_vars[8] = 0
 Fraction_lower_bound_9: Fraction[9] >= 0
 Fraction_upper_bound_9: Fraction[9] <= 1
 R92: 0.7340851174744587 alphas[0] + 0.6887531183124831 alphas[1]
   - 1.460925496263445 alphas[2] + 1.29617249954311 alphas[3] - alphas[4]
   + f_var[9] = 0
 R93: - 2053.400566507637 Fraction[9] + Q0_var[9] = 0
 R94: Q0_var[9] + Lost_0_expr_9 = 2226.5126428821
 R95: Q0_var[9] + Q_hat_adjusted_9 = 2802.887150206102
 R96: Sold_0[9] - Q0_var[9] - Q1_var[9] + total_stock_second_period_9 = 0
 R97: total_stock_second_period_9 + Lost_1_expr_9 = 576.3745057584925
 R98: Sold_1[9] + Left_1[9] - total_stock_second_period_9 = 0
 R99: - 500 Sold_0[9] - 500 Sold_1[9] + 500 Lost_0[9] + 500 Lost_1[9]
   + 500 Left_1[9] + profits_vars[9] = 0
 Fraction_lower_bound_10: Fraction[10] >= 0
 Fraction_upper_bound_10: Fraction[10] <= 1
 R102: - 0.7115448711071122 alphas[0] + 0.9704687421482601 alphas[1]
   + 0.7855052919242557 alphas[2] - 0.6760262272279475 alphas[3]
   - alphas[4] + f_var[10] = 0
 R103: - 2053.400566507637 Fraction[10] + Q0_var[10] = 0
 R104: Q0_var[10] + Lost_0_expr_10 = 1689.973234955106
 R105: Q0_var[10] + Q_hat_adjusted_10 = 2013.535923713845
 R106: Sold_0[10] - Q0_var[10] - Q1_var[10] + total_stock_second_period_10
   = 0
 R107: total_stock_second_period_10 + Lost_1_expr_10 = 323.5626893088137
 R108: Sold_1[10] + Left_1[10] - total_stock_second_period_10 = 0
 R109: - 500 Sold_0[10] - 500 Sold_1[10] + 500 Lost_0[10] + 500 Lost_1[10]
   + 500 Left_1[10] + profits_vars[10] = 0
 Fraction_lower_bound_11: Fraction[11] >= 0
 Fraction_upper_bound_11: Fraction[11] <= 1
 R112: 0.2190709461845003 alphas[0] + 1.130665947024758 alphas[1]
   - 0.9884669399077892 alphas[2] + 0.0895652096157003 alphas[3]
   - alphas[4] + f_var[11] = 0
 R113: - 2053.400566507637 Fraction[11] + Q0_var[11] = 0
 R114: Q0_var[11] + Lost_0_expr_11 = 2308.46269322693
 R115: Q0_var[11] + Q_hat_adjusted_11 = 2831.667027060258
 R116: Sold_0[11] - Q0_var[11] - Q1_var[11] + total_stock_second_period_11
   = 0
 R117: total_stock_second_period_11 + Lost_1_expr_11 = 523.2043362355639
 R118: Sold_1[11] + Left_1[11] - total_stock_second_period_11 = 0
 R119: - 500 Sold_0[11] - 500 Sold_1[11] + 500 Lost_0[11] + 500 Lost_1[11]
   + 500 Left_1[11] + profits_vars[11] = 0
 Fraction_lower_bound_12: Fraction[12] >= 0
 Fraction_upper_bound_12: Fraction[12] <= 1
 R122: 0.0804518423132196 alphas[0] - 0.6691360908629208 alphas[1]
   - 0.3356756145437382 alphas[2] - 0.7533331671833704 alphas[3]
   - alphas[4] + f_var[12] = 0
 R123: - 2053.400566507637 Fraction[12] + Q0_var[12] = 0
 R124: Q0_var[12] + Lost_0_expr_12 = 1875.930050515312
 R125: Q0_var[12] + Q_hat_adjusted_12 = 2325.669687732558
 R126: Sold_0[12] - Q0_var[12] - Q1_var[12] + total_stock_second_period_12
   = 0
 R127: total_stock_second_period_12 + Lost_1_expr_12 = 449.7396376587521
 R128: Sold_1[12] + Left_1[12] - total_stock_second_period_12 = 0
 R129: - 500 Sold_0[12] - 500 Sold_1[12] + 500 Lost_0[12] + 500 Lost_1[12]
   + 500 Left_1[12] + profits_vars[12] = 0
 Fraction_lower_bound_13: Fraction[13] >= 0
 Fraction_upper_bound_13: Fraction[13] <= 1
 R132: - 1.185550778167024 alphas[0] + 1.03899428175183 alphas[1]
   + 1.450554690095896 alphas[2] - 1.206394289581885 alphas[3] - alphas[4]
   + f_var[13] = 0
 R133: - 2053.400566507637 Fraction[13] + Q0_var[13] = 0
 R134: Q0_var[13] + Lost_0_expr_13 = 1094.869092034218
 R135: Q0_var[13] + Q_hat_adjusted_13 = 1343.587567901272
 R136: Sold_0[13] - Q0_var[13] - Q1_var[13] + total_stock_second_period_13
   = 0
 R137: total_stock_second_period_13 + Lost_1_expr_13 = 248.7184754275668
 R138: Sold_1[13] + Left_1[13] - total_stock_second_period_13 = 0
 R139: - 500 Sold_0[13] - 500 Sold_1[13] + 500 Lost_0[13] + 500 Lost_1[13]
   + 500 Left_1[13] + profits_vars[13] = 0
 Fraction_lower_bound_14: Fraction[14] >= 0
 Fraction_upper_bound_14: Fraction[14] <= 1
 R142: 1.840234842308595 alphas[0] + 0.8464738653451827 alphas[1]
   - 0.8435804691776236 alphas[2] - 1.548996939710779 alphas[3] - alphas[4]
   + f_var[14] = 0
 R143: - 2053.400566507637 Fraction[14] + Q0_var[14] = 0
 R144: Q0_var[14] + Lost_0_expr_14 = 1927.073132752433
 R145: Q0_var[14] + Q_hat_adjusted_14 = 2433.972044150916
 R146: Sold_0[14] - Q0_var[14] - Q1_var[14] + total_stock_second_period_14
   = 0
 R147: total_stock_second_period_14 + Lost_1_expr_14 = 506.8989098512386
 R148: Sold_1[14] + Left_1[14] - total_stock_second_period_14 = 0
 R149: - 500 Sold_0[14] - 500 Sold_1[14] + 500 Lost_0[14] + 500 Lost_1[14]
   + 500 Left_1[14] + profits_vars[14] = 0
 Fraction_lower_bound_15: Fraction[15] >= 0
 Fraction_upper_bound_15: Fraction[15] <= 1
 R152: 1.783252692812304 alphas[0] + 0.2789851684846042 alphas[1]
   + 1.010402835029238 alphas[2] - 1.173185428878948 alphas[3] - alphas[4]
   + f_var[15] = 0
 R153: - 2053.400566507637 Fraction[15] + Q0_var[15] = 0
 R154: Q0_var[15] + Lost_0_expr_15 = 1206.0836216997
 R155: Q0_var[15] + Q_hat_adjusted_15 = 1504.336492944726
 R156: Sold_0[15] - Q0_var[15] - Q1_var[15] + total_stock_second_period_15
   = 0
 R157: total_stock_second_period_15 + Lost_1_expr_15 = 298.2528704234923
 R158: Sold_1[15] + Left_1[15] - total_stock_second_period_15 = 0
 R159: - 500 Sold_0[15] - 500 Sold_1[15] + 500 Lost_0[15] + 500 Lost_1[15]
   + 500 Left_1[15] + profits_vars[15] = 0
 Fraction_lower_bound_16: Fraction[16] >= 0
 Fraction_upper_bound_16: Fraction[16] <= 1
 R162: 2.020167489294702 alphas[0] - 1.211968842659354 alphas[1]
   - 1.142610938055166 alphas[2] + 1.476524961653682 alphas[3] - alphas[4]
   + f_var[16] = 0
 R163: - 2053.400566507637 Fraction[16] + Q0_var[16] = 0
 R164: Q0_var[16] + Lost_0_expr_16 = 1906.026838820684
 R165: Q0_var[16] + Q_hat_adjusted_16 = 2446.578434794819
 R166: Sold_0[16] - Q0_var[16] - Q1_var[16] + total_stock_second_period_16
   = 0
 R167: total_stock_second_period_16 + Lost_1_expr_16 = 540.5515985555808
 R168: Sold_1[16] + Left_1[16] - total_stock_second_period_16 = 0
 R169: - 500 Sold_0[16] - 500 Sold_1[16] + 500 Lost_0[16] + 500 Lost_1[16]
   + 500 Left_1[16] + profits_vars[16] = 0
 Fraction_lower_bound_17: Fraction[17] >= 0
 Fraction_upper_bound_17: Fraction[17] <= 1
 R172: - 0.8563431724362236 alphas[0] + 1.174424780788562 alphas[1]
   + 0.1895025714961081 alphas[2] + 0.3832772058522401 alphas[3]
   - alphas[4] + f_var[17] = 0
 R173: - 2053.400566507637 Fraction[17] + Q0_var[17] = 0
 R174: Q0_var[17] + Lost_0_expr_17 = 1396.008966051397
 R175: Q0_var[17] + Q_hat_adjusted_17 = 1786.645402830598
 R176: Sold_0[17] - Q0_var[17] - Q1_var[17] + total_stock_second_period_17
   = 0
 R177: total_stock_second_period_17 + Lost_1_expr_17 = 390.6364363179293
 R178: Sold_1[17] + Left_1[17] - total_stock_second_period_17 = 0
 R179: - 500 Sold_0[17] - 500 Sold_1[17] + 500 Lost_0[17] + 500 Lost_1[17]
   + 500 Left_1[17] + profits_vars[17] = 0
 Fraction_lower_bound_18: Fraction[18] >= 0
 Fraction_upper_bound_18: Fraction[18] <= 1
 R182: - 0.6635029545277169 alphas[0] - 1.267852010342292 alphas[1]
   - 1.145419577230695 alphas[2] - 0.7782420895307338 alphas[3] - alphas[4]
   + f_var[18] = 0
 R183: - 2053.400566507637 Fraction[18] + Q0_var[18] = 0
 R184: Q0_var[18] + Lost_0_expr_18 = 2250.503717816093
 R185: Q0_var[18] + Q_hat_adjusted_18 = 2791.371400639382
 R186: Sold_0[18] - Q0_var[18] - Q1_var[18] + total_stock_second_period_18
   = 0
 R187: total_stock_second_period_18 + Lost_1_expr_18 = 540.8676809274264
 R188: Sold_1[18] + Left_1[18] - total_stock_second_period_18 = 0
 R189: - 500 Sold_0[18] - 500 Sold_1[18] + 500 Lost_0[18] + 500 Lost_1[18]
   + 500 Left_1[18] + profits_vars[18] = 0
 Fraction_lower_bound_19: Fraction[19] >= 0
 Fraction_upper_bound_19: Fraction[19] <= 1
 R192: - 0.9887399790034055 alphas[0] + 1.177730850908782 alphas[1]
   - 0.3048078049445529 alphas[2] + 0.9745282776980045 alphas[3]
   - alphas[4] + f_var[19] = 0
 R193: - 2053.400566507637 Fraction[19] + Q0_var[19] = 0
 R194: Q0_var[19] + Lost_0_expr_19 = 1703.322760250565
 R195: Q0_var[19] + Q_hat_adjusted_19 = 2149.588554172074
 R196: Sold_0[19] - Q0_var[19] - Q1_var[19] + total_stock_second_period_19
   = 0
 R197: total_stock_second_period_19 + Lost_1_expr_19 = 446.2657950388829
 R198: Sold_1[19] + Left_1[19] - total_stock_second_period_19 = 0
 R199: - 500 Sold_0[19] - 500 Sold_1[19] + 500 Lost_0[19] + 500 Lost_1[19]
   + 500 Left_1[19] + profits_vars[19] = 0
 Fraction_lower_bound_20: Fraction[20] >= 0
 Fraction_upper_bound_20: Fraction[20] <= 1
 R202: - 1.373287391687492 alphas[0] - 1.724481283476159 alphas[1]
   - 0.450191155178684 alphas[2] - 0.1223073485596939 alphas[3] - alphas[4]
   + f_var[20] = 0
 R203: - 2053.400566507637 Fraction[20] + Q0_var[20] = 0
 R204: Q0_var[20] + Lost_0_expr_20 = 1667.939937847112
 R205: Q0_var[20] + Q_hat_adjusted_20 = 2130.567076467601
 R206: Sold_0[20] - Q0_var[20] - Q1_var[20] + total_stock_second_period_20
   = 0
 R207: total_stock_second_period_20 + Lost_1_expr_20 = 462.6271399098202
 R208: Sold_1[20] + Left_1[20] - total_stock_second_period_20 = 0
 R209: - 500 Sold_0[20] - 500 Sold_1[20] + 500 Lost_0[20] + 500 Lost_1[20]
   + 500 Left_1[20] + profits_vars[20] = 0
 Fraction_lower_bound_21: Fraction[21] >= 0
 Fraction_upper_bound_21: Fraction[21] <= 1
 R212: - 0.7378651309105446 alphas[0] - 0.0504325599864221 alphas[1]
   + 0.5447050171735942 alphas[2] + 1.342722759310279 alphas[3] - alphas[4]
   + f_var[21] = 0
 R213: - 2053.400566507637 Fraction[21] + Q0_var[21] = 0
 R214: Q0_var[21] + Lost_0_expr_21 = 1254.414601510171
 R215: Q0_var[21] + Q_hat_adjusted_21 = 1605.076793335005
 R216: Sold_0[21] - Q0_var[21] - Q1_var[21] + total_stock_second_period_21
   = 0
 R217: total_stock_second_period_21 + Lost_1_expr_21 = 350.6621909633511
 R218: Sold_1[21] + Left_1[21] - total_stock_second_period_21 = 0
 R219: - 500 Sold_0[21] - 500 Sold_1[21] + 500 Lost_0[21] + 500 Lost_1[21]
   + 500 Left_1[21] + profits_vars[21] = 0
 Fraction_lower_bound_22: Fraction[22] >= 0
 Fraction_upper_bound_22: Fraction[22] <= 1
 R222: 0.4577726008491022 alphas[0] - 1.725475619194501 alphas[1]
   - 1.475740864355606 alphas[2] + 0.8854809048116157 alphas[3] - alphas[4]
   + f_var[22] = 0
 R223: - 2053.400566507637 Fraction[22] + Q0_var[22] = 0
 R224: Q0_var[22] + Lost_0_expr_22 = 2456.134222796946
 R225: Q0_var[22] + Q_hat_adjusted_22 = 3034.176040048083
 R226: Sold_0[22] - Q0_var[22] - Q1_var[22] + total_stock_second_period_22
   = 0
 R227: total_stock_second_period_22 + Lost_1_expr_22 = 578.0418173616119
 R228: Sold_1[22] + Left_1[22] - total_stock_second_period_22 = 0
 R229: - 500 Sold_0[22] - 500 Sold_1[22] + 500 Lost_0[22] + 500 Lost_1[22]
   + 500 Left_1[22] + profits_vars[22] = 0
 Fraction_lower_bound_23: Fraction[23] >= 0
 Fraction_upper_bound_23: Fraction[23] <= 1
 R232: - 0.6719031202226472 alphas[0] - 0.4994128952930674 alphas[1]
   - 0.0891436867992024 alphas[2] + 1.455154637985153 alphas[3] - alphas[4]
   + f_var[23] = 0
 R233: - 2053.400566507637 Fraction[23] + Q0_var[23] = 0
 R234: Q0_var[23] + Lost_0_expr_23 = 1837.484567604658
 R235: Q0_var[23] + Q_hat_adjusted_23 = 2259.4796675568
 R236: Sold_0[23] - Q0_var[23] - Q1_var[23] + total_stock_second_period_23
   = 0
 R237: total_stock_second_period_23 + Lost_1_expr_23 = 421.9950996148187
 R238: Sold_1[23] + Left_1[23] - total_stock_second_period_23 = 0
 R239: - 500 Sold_0[23] - 500 Sold_1[23] + 500 Lost_0[23] + 500 Lost_1[23]
   + 500 Left_1[23] + profits_vars[23] = 0
 Fraction_lower_bound_24: Fraction[24] >= 0
 Fraction_upper_bound_24: Fraction[24] <= 1
 R242: 1.67297557835104 alphas[0] - 0.9425375890211247 alphas[1]
   + 0.888892836523537 alphas[2] - 0.9782274247791758 alphas[3] - alphas[4]
   + f_var[24] = 0
 R243: - 2053.400566507637 Fraction[24] + Q0_var[24] = 0
 R244: Q0_var[24] + Lost_0_expr_24 = 1280.235565652873
 R245: Q0_var[24] + Q_hat_adjusted_24 = 1592.163090652771
 R246: Sold_0[24] - Q0_var[24] - Q1_var[24] + total_stock_second_period_24
   = 0
 R247: total_stock_second_period_24 + Lost_1_expr_24 = 311.9275242790899
 R248: Sold_1[24] + Left_1[24] - total_stock_second_period_24 = 0
 R249: - 500 Sold_0[24] - 500 Sold_1[24] + 500 Lost_0[24] + 500 Lost_1[24]
   + 500 Left_1[24] + profits_vars[24] = 0
Bounds
 alphas[0] free
 alphas[1] free
 alphas[2] free
 alphas[3] free
 alphas[4] free
 f_var[0] free
 f_var[1] free
 f_var[2] free
 f_var[3] free
 f_var[4] free
 f_var[5] free
 f_var[6] free
 f_var[7] free
 f_var[8] free
 f_var[9] free
 f_var[10] free
 f_var[11] free
 f_var[12] free
 f_var[13] free
 f_var[14] free
 f_var[15] free
 f_var[16] free
 f_var[17] free
 f_var[18] free
 f_var[19] free
 f_var[20] free
 f_var[21] free
 f_var[22] free
 f_var[23] free
 f_var[24] free
 Fraction[0] <= 1
 Fraction[1] <= 1
 Fraction[2] <= 1
 Fraction[3] <= 1
 Fraction[4] <= 1
 Fraction[5] <= 1
 Fraction[6] <= 1
 Fraction[7] <= 1
 Fraction[8] <= 1
 Fraction[9] <= 1
 Fraction[10] <= 1
 Fraction[11] <= 1
 Fraction[12] <= 1
 Fraction[13] <= 1
 Fraction[14] <= 1
 Fraction[15] <= 1
 Fraction[16] <= 1
 Fraction[17] <= 1
 Fraction[18] <= 1
 Fraction[19] <= 1
 Fraction[20] <= 1
 Fraction[21] <= 1
 Fraction[22] <= 1
 Fraction[23] <= 1
 Fraction[24] <= 1
 Q0_var[0] <= 2054.400566507637
 Q0_var[1] <= 2054.400566507637
 Q0_var[2] <= 2054.400566507637
 Q0_var[3] <= 2054.400566507637
 Q0_var[4] <= 2054.400566507637
 Q0_var[5] <= 2054.400566507637
 Q0_var[6] <= 2054.400566507637
 Q0_var[7] <= 2054.400566507637
 Q0_var[8] <= 2054.400566507637
 Q0_var[9] <= 2054.400566507637
 Q0_var[10] <= 2054.400566507637
 Q0_var[11] <= 2054.400566507637
 Q0_var[12] <= 2054.400566507637
 Q0_var[13] <= 2054.400566507637
 Q0_var[14] <= 2054.400566507637
 Q0_var[15] <= 2054.400566507637
 Q0_var[16] <= 2054.400566507637
 Q0_var[17] <= 2054.400566507637
 Q0_var[18] <= 2054.400566507637
 Q0_var[19] <= 2054.400566507637
 Q0_var[20] <= 2054.400566507637
 Q0_var[21] <= 2054.400566507637
 Q0_var[22] <= 2054.400566507637
 Q0_var[23] <= 2054.400566507637
 Q0_var[24] <= 2054.400566507637
 profits_vars[0] free
 profits_vars[1] free
 profits_vars[2] free
 profits_vars[3] free
 profits_vars[4] free
 profits_vars[5] free
 profits_vars[6] free
 profits_vars[7] free
 profits_vars[8] free
 profits_vars[9] free
 profits_vars[10] free
 profits_vars[11] free
 profits_vars[12] free
 profits_vars[13] free
 profits_vars[14] free
 profits_vars[15] free
 profits_vars[16] free
 profits_vars[17] free
 profits_vars[18] free
 profits_vars[19] free
 profits_vars[20] free
 profits_vars[21] free
 profits_vars[22] free
 profits_vars[23] free
 profits_vars[24] free
 Lost_0_expr_0 free
 Q_hat_adjusted_0 free
 Lost_1_expr_0 free
 Lost_0_expr_1 free
 Q_hat_adjusted_1 free
 Lost_1_expr_1 free
 Lost_0_expr_2 free
 Q_hat_adjusted_2 free
 Lost_1_expr_2 free
 Lost_0_expr_3 free
 Q_hat_adjusted_3 free
 Lost_1_expr_3 free
 Lost_0_expr_4 free
 Q_hat_adjusted_4 free
 Lost_1_expr_4 free
 Lost_0_expr_5 free
 Q_hat_adjusted_5 free
 Lost_1_expr_5 free
 Lost_0_expr_6 free
 Q_hat_adjusted_6 free
 Lost_1_expr_6 free
 Lost_0_expr_7 free
 Q_hat_adjusted_7 free
 Lost_1_expr_7 free
 Lost_0_expr_8 free
 Q_hat_adjusted_8 free
 Lost_1_expr_8 free
 Lost_0_expr_9 free
 Q_hat_adjusted_9 free
 Lost_1_expr_9 free
 Lost_0_expr_10 free
 Q_hat_adjusted_10 free
 Lost_1_expr_10 free
 Lost_0_expr_11 free
 Q_hat_adjusted_11 free
 Lost_1_expr_11 free
 Lost_0_expr_12 free
 Q_hat_adjusted_12 free
 Lost_1_expr_12 free
 Lost_0_expr_13 free
 Q_hat_adjusted_13 free
 Lost_1_expr_13 free
 Lost_0_expr_14 free
 Q_hat_adjusted_14 free
 Lost_1_expr_14 free
 Lost_0_expr_15 free
 Q_hat_adjusted_15 free
 Lost_1_expr_15 free
 Lost_0_expr_16 free
 Q_hat_adjusted_16 free
 Lost_1_expr_16 free
 Lost_0_expr_17 free
 Q_hat_adjusted_17 free
 Lost_1_expr_17 free
 Lost_0_expr_18 free
 Q_hat_adjusted_18 free
 Lost_1_expr_18 free
 Lost_0_expr_19 free
 Q_hat_adjusted_19 free
 Lost_1_expr_19 free
 Lost_0_expr_20 free
 Q_hat_adjusted_20 free
 Lost_1_expr_20 free
 Lost_0_expr_21 free
 Q_hat_adjusted_21 free
 Lost_1_expr_21 free
 Lost_0_expr_22 free
 Q_hat_adjusted_22 free
 Lost_1_expr_22 free
 Lost_0_expr_23 free
 Q_hat_adjusted_23 free
 Lost_1_expr_23 free
 Lost_0_expr_24 free
 Q_hat_adjusted_24 free
 Lost_1_expr_24 free
General Constraints
 GC0: Fraction[0] = LOGISTIC ( f_var[0] )
 min_sales_constr_0: Sold_0[0] = MIN ( Q0_var[0] , 1626.043314427274 )
 max_lost_constr_0: Lost_0[0] = MAX ( Lost_0_expr_0 , 0 )
 max_Q1_constr_0: Q1_var[0] = MAX ( Q_hat_adjusted_0 , 0 )
 min_sales2_constr_0: Sold_1[0] = MIN ( total_stock_second_period_0
   , 438.9082683981151 )
 max_lost2_constr_0: Lost_1[0] = MAX ( Lost_1_expr_0 , 0 )
 GC6: Fraction[1] = LOGISTIC ( f_var[1] )
 min_sales_constr_1: Sold_0[1] = MIN ( Q0_var[1] , 739.4991742575022 )
 max_lost_constr_1: Lost_0[1] = MAX ( Lost_0_expr_1 , 0 )
 max_Q1_constr_1: Q1_var[1] = MAX ( Q_hat_adjusted_1 , 0 )
 min_sales2_constr_1: Sold_1[1] = MIN ( total_stock_second_period_1
   , 235.0039865960826 )
 max_lost2_constr_1: Lost_1[1] = MAX ( Lost_1_expr_1 , 0 )
 GC12: Fraction[2] = LOGISTIC ( f_var[2] )
 min_sales_constr_2: Sold_0[2] = MIN ( Q0_var[2] , 1598.520274925257 )
 max_lost_constr_2: Lost_0[2] = MAX ( Lost_0_expr_2 , 0 )
 max_Q1_constr_2: Q1_var[2] = MAX ( Q_hat_adjusted_2 , 0 )
 min_sales2_constr_2: Sold_1[2] = MIN ( total_stock_second_period_2
   , 467.5970110612975 )
 max_lost2_constr_2: Lost_1[2] = MAX ( Lost_1_expr_2 , 0 )
 GC18: Fraction[3] = LOGISTIC ( f_var[3] )
 min_sales_constr_3: Sold_0[3] = MIN ( Q0_var[3] , 2398.802338453156 )
 max_lost_constr_3: Lost_0[3] = MAX ( Lost_0_expr_3 , 0 )
 max_Q1_constr_3: Q1_var[3] = MAX ( Q_hat_adjusted_3 , 0 )
 min_sales2_constr_3: Sold_1[3] = MIN ( total_stock_second_period_3
   , 581.0942725587191 )
 max_lost2_constr_3: Lost_1[3] = MAX ( Lost_1_expr_3 , 0 )
 GC24: Fraction[4] = LOGISTIC ( f_var[4] )
 min_sales_constr_4: Sold_0[4] = MIN ( Q0_var[4] , 1363.144551235566 )
 max_lost_constr_4: Lost_0[4] = MAX ( Lost_0_expr_4 , 0 )
 max_Q1_constr_4: Q1_var[4] = MAX ( Q_hat_adjusted_4 , 0 )
 min_sales2_constr_4: Sold_1[4] = MIN ( total_stock_second_period_4
   , 224.376571759979 )
 max_lost2_constr_4: Lost_1[4] = MAX ( Lost_1_expr_4 , 0 )
 GC30: Fraction[5] = LOGISTIC ( f_var[5] )
 min_sales_constr_5: Sold_0[5] = MIN ( Q0_var[5] , 1567.672756172823 )
 max_lost_constr_5: Lost_0[5] = MAX ( Lost_0_expr_5 , 0 )
 max_Q1_constr_5: Q1_var[5] = MAX ( Q_hat_adjusted_5 , 0 )
 min_sales2_constr_5: Sold_1[5] = MIN ( total_stock_second_period_5
   , 388.0786672089184 )
 max_lost2_constr_5: Lost_1[5] = MAX ( Lost_1_expr_5 , 0 )
 GC36: Fraction[6] = LOGISTIC ( f_var[6] )
 min_sales_constr_6: Sold_0[6] = MIN ( Q0_var[6] , 1672.528001503375 )
 max_lost_constr_6: Lost_0[6] = MAX ( Lost_0_expr_6 , 0 )
 max_Q1_constr_6: Q1_var[6] = MAX ( Q_hat_adjusted_6 , 0 )
 min_sales2_constr_6: Sold_1[6] = MIN ( total_stock_second_period_6
   , 396.0209656359195 )
 max_lost2_constr_6: Lost_1[6] = MAX ( Lost_1_expr_6 , 0 )
 GC42: Fraction[7] = LOGISTIC ( f_var[7] )
 min_sales_constr_7: Sold_0[7] = MIN ( Q0_var[7] , 1279.491825681141 )
 max_lost_constr_7: Lost_0[7] = MAX ( Lost_0_expr_7 , 0 )
 max_Q1_constr_7: Q1_var[7] = MAX ( Q_hat_adjusted_7 , 0 )
 min_sales2_constr_7: Sold_1[7] = MIN ( total_stock_second_period_7
   , 386.1259528954367 )
 max_lost2_constr_7: Lost_1[7] = MAX ( Lost_1_expr_7 , 0 )
 GC48: Fraction[8] = LOGISTIC ( f_var[8] )
 min_sales_constr_8: Sold_0[8] = MIN ( Q0_var[8] , 709.2630595513233 )
 max_lost_constr_8: Lost_0[8] = MAX ( Lost_0_expr_8 , 0 )
 max_Q1_constr_8: Q1_var[8] = MAX ( Q_hat_adjusted_8 , 0 )
 min_sales2_constr_8: Sold_1[8] = MIN ( total_stock_second_period_8
   , 211.5408163203091 )
 max_lost2_constr_8: Lost_1[8] = MAX ( Lost_1_expr_8 , 0 )
 GC54: Fraction[9] = LOGISTIC ( f_var[9] )
 min_sales_constr_9: Sold_0[9] = MIN ( Q0_var[9] , 2226.5126428821 )
 max_lost_constr_9: Lost_0[9] = MAX ( Lost_0_expr_9 , 0 )
 max_Q1_constr_9: Q1_var[9] = MAX ( Q_hat_adjusted_9 , 0 )
 min_sales2_constr_9: Sold_1[9] = MIN ( total_stock_second_period_9
   , 576.3745057584925 )
 max_lost2_constr_9: Lost_1[9] = MAX ( Lost_1_expr_9 , 0 )
 GC60: Fraction[10] = LOGISTIC ( f_var[10] )
 min_sales_constr_10: Sold_0[10] = MIN ( Q0_var[10] , 1689.973234955106 )
 max_lost_constr_10: Lost_0[10] = MAX ( Lost_0_expr_10 , 0 )
 max_Q1_constr_10: Q1_var[10] = MAX ( Q_hat_adjusted_10 , 0 )
 min_sales2_constr_10: Sold_1[10] = MIN ( total_stock_second_period_10
   , 323.5626893088137 )
 max_lost2_constr_10: Lost_1[10] = MAX ( Lost_1_expr_10 , 0 )
 GC66: Fraction[11] = LOGISTIC ( f_var[11] )
 min_sales_constr_11: Sold_0[11] = MIN ( Q0_var[11] , 2308.46269322693 )
 max_lost_constr_11: Lost_0[11] = MAX ( Lost_0_expr_11 , 0 )
 max_Q1_constr_11: Q1_var[11] = MAX ( Q_hat_adjusted_11 , 0 )
 min_sales2_constr_11: Sold_1[11] = MIN ( total_stock_second_period_11
   , 523.2043362355639 )
 max_lost2_constr_11: Lost_1[11] = MAX ( Lost_1_expr_11 , 0 )
 GC72: Fraction[12] = LOGISTIC ( f_var[12] )
 min_sales_constr_12: Sold_0[12] = MIN ( Q0_var[12] , 1875.930050515312 )
 max_lost_constr_12: Lost_0[12] = MAX ( Lost_0_expr_12 , 0 )
 max_Q1_constr_12: Q1_var[12] = MAX ( Q_hat_adjusted_12 , 0 )
 min_sales2_constr_12: Sold_1[12] = MIN ( total_stock_second_period_12
   , 449.7396376587521 )
 max_lost2_constr_12: Lost_1[12] = MAX ( Lost_1_expr_12 , 0 )
 GC78: Fraction[13] = LOGISTIC ( f_var[13] )
 min_sales_constr_13: Sold_0[13] = MIN ( Q0_var[13] , 1094.869092034218 )
 max_lost_constr_13: Lost_0[13] = MAX ( Lost_0_expr_13 , 0 )
 max_Q1_constr_13: Q1_var[13] = MAX ( Q_hat_adjusted_13 , 0 )
 min_sales2_constr_13: Sold_1[13] = MIN ( total_stock_second_period_13
   , 248.7184754275668 )
 max_lost2_constr_13: Lost_1[13] = MAX ( Lost_1_expr_13 , 0 )
 GC84: Fraction[14] = LOGISTIC ( f_var[14] )
 min_sales_constr_14: Sold_0[14] = MIN ( Q0_var[14] , 1927.073132752433 )
 max_lost_constr_14: Lost_0[14] = MAX ( Lost_0_expr_14 , 0 )
 max_Q1_constr_14: Q1_var[14] = MAX ( Q_hat_adjusted_14 , 0 )
 min_sales2_constr_14: Sold_1[14] = MIN ( total_stock_second_period_14
   , 506.8989098512386 )
 max_lost2_constr_14: Lost_1[14] = MAX ( Lost_1_expr_14 , 0 )
 GC90: Fraction[15] = LOGISTIC ( f_var[15] )
 min_sales_constr_15: Sold_0[15] = MIN ( Q0_var[15] , 1206.0836216997 )
 max_lost_constr_15: Lost_0[15] = MAX ( Lost_0_expr_15 , 0 )
 max_Q1_constr_15: Q1_var[15] = MAX ( Q_hat_adjusted_15 , 0 )
 min_sales2_constr_15: Sold_1[15] = MIN ( total_stock_second_period_15
   , 298.2528704234923 )
 max_lost2_constr_15: Lost_1[15] = MAX ( Lost_1_expr_15 , 0 )
 GC96: Fraction[16] = LOGISTIC ( f_var[16] )
 min_sales_constr_16: Sold_0[16] = MIN ( Q0_var[16] , 1906.026838820684 )
 max_lost_constr_16: Lost_0[16] = MAX ( Lost_0_expr_16 , 0 )
 max_Q1_constr_16: Q1_var[16] = MAX ( Q_hat_adjusted_16 , 0 )
 min_sales2_constr_16: Sold_1[16] = MIN ( total_stock_second_period_16
   , 540.5515985555808 )
 max_lost2_constr_16: Lost_1[16] = MAX ( Lost_1_expr_16 , 0 )
 GC102: Fraction[17] = LOGISTIC ( f_var[17] )
 min_sales_constr_17: Sold_0[17] = MIN ( Q0_var[17] , 1396.008966051397 )
 max_lost_constr_17: Lost_0[17] = MAX ( Lost_0_expr_17 , 0 )
 max_Q1_constr_17: Q1_var[17] = MAX ( Q_hat_adjusted_17 , 0 )
 min_sales2_constr_17: Sold_1[17] = MIN ( total_stock_second_period_17
   , 390.6364363179293 )
 max_lost2_constr_17: Lost_1[17] = MAX ( Lost_1_expr_17 , 0 )
 GC108: Fraction[18] = LOGISTIC ( f_var[18] )
 min_sales_constr_18: Sold_0[18] = MIN ( Q0_var[18] , 2250.503717816093 )
 max_lost_constr_18: Lost_0[18] = MAX ( Lost_0_expr_18 , 0 )
 max_Q1_constr_18: Q1_var[18] = MAX ( Q_hat_adjusted_18 , 0 )
 min_sales2_constr_18: Sold_1[18] = MIN ( total_stock_second_period_18
   , 540.8676809274264 )
 max_lost2_constr_18: Lost_1[18] = MAX ( Lost_1_expr_18 , 0 )
 GC114: Fraction[19] = LOGISTIC ( f_var[19] )
 min_sales_constr_19: Sold_0[19] = MIN ( Q0_var[19] , 1703.322760250565 )
 max_lost_constr_19: Lost_0[19] = MAX ( Lost_0_expr_19 , 0 )
 max_Q1_constr_19: Q1_var[19] = MAX ( Q_hat_adjusted_19 , 0 )
 min_sales2_constr_19: Sold_1[19] = MIN ( total_stock_second_period_19
   , 446.2657950388829 )
 max_lost2_constr_19: Lost_1[19] = MAX ( Lost_1_expr_19 , 0 )
 GC120: Fraction[20] = LOGISTIC ( f_var[20] )
 min_sales_constr_20: Sold_0[20] = MIN ( Q0_var[20] , 1667.939937847112 )
 max_lost_constr_20: Lost_0[20] = MAX ( Lost_0_expr_20 , 0 )
 max_Q1_constr_20: Q1_var[20] = MAX ( Q_hat_adjusted_20 , 0 )
 min_sales2_constr_20: Sold_1[20] = MIN ( total_stock_second_period_20
   , 462.6271399098202 )
 max_lost2_constr_20: Lost_1[20] = MAX ( Lost_1_expr_20 , 0 )
 GC126: Fraction[21] = LOGISTIC ( f_var[21] )
 min_sales_constr_21: Sold_0[21] = MIN ( Q0_var[21] , 1254.414601510171 )
 max_lost_constr_21: Lost_0[21] = MAX ( Lost_0_expr_21 , 0 )
 max_Q1_constr_21: Q1_var[21] = MAX ( Q_hat_adjusted_21 , 0 )
 min_sales2_constr_21: Sold_1[21] = MIN ( total_stock_second_period_21
   , 350.6621909633511 )
 max_lost2_constr_21: Lost_1[21] = MAX ( Lost_1_expr_21 , 0 )
 GC132: Fraction[22] = LOGISTIC ( f_var[22] )
 min_sales_constr_22: Sold_0[22] = MIN ( Q0_var[22] , 2456.134222796946 )
 max_lost_constr_22: Lost_0[22] = MAX ( Lost_0_expr_22 , 0 )
 max_Q1_constr_22: Q1_var[22] = MAX ( Q_hat_adjusted_22 , 0 )
 min_sales2_constr_22: Sold_1[22] = MIN ( total_stock_second_period_22
   , 578.0418173616119 )
 max_lost2_constr_22: Lost_1[22] = MAX ( Lost_1_expr_22 , 0 )
 GC138: Fraction[23] = LOGISTIC ( f_var[23] )
 min_sales_constr_23: Sold_0[23] = MIN ( Q0_var[23] , 1837.484567604658 )
 max_lost_constr_23: Lost_0[23] = MAX ( Lost_0_expr_23 , 0 )
 max_Q1_constr_23: Q1_var[23] = MAX ( Q_hat_adjusted_23 , 0 )
 min_sales2_constr_23: Sold_1[23] = MIN ( total_stock_second_period_23
   , 421.9950996148187 )
 max_lost2_constr_23: Lost_1[23] = MAX ( Lost_1_expr_23 , 0 )
 GC144: Fraction[24] = LOGISTIC ( f_var[24] )
 min_sales_constr_24: Sold_0[24] = MIN ( Q0_var[24] , 1280.235565652873 )
 max_lost_constr_24: Lost_0[24] = MAX ( Lost_0_expr_24 , 0 )
 max_Q1_constr_24: Q1_var[24] = MAX ( Q_hat_adjusted_24 , 0 )
 min_sales2_constr_24: Sold_1[24] = MIN ( total_stock_second_period_24
   , 311.9275242790899 )
 max_lost2_constr_24: Lost_1[24] = MAX ( Lost_1_expr_24 , 0 )
End
