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
   + profits_vars[23] + profits_vars[24] + profits_vars[25]
   + profits_vars[26] + profits_vars[27] + profits_vars[28]
   + profits_vars[29] + profits_vars[30] + profits_vars[31]
   + profits_vars[32] + profits_vars[33] + profits_vars[34]
   + profits_vars[35] + profits_vars[36] + profits_vars[37]
   + profits_vars[38] + profits_vars[39] + profits_vars[40]
   + profits_vars[41] + profits_vars[42] + profits_vars[43]
   + profits_vars[44] + profits_vars[45] + profits_vars[46]
   + profits_vars[47] + profits_vars[48] + profits_vars[49]
Subject To
 Fraction_lower_bound_0: Fraction[0] >= 0
 Fraction_upper_bound_0: Fraction[0] <= 1
 R2: - 0.0398448308177865 alphas[0] - 0.3779089760840579 alphas[1]
   + 0.5400971290116704 alphas[2] - 1.778249581914427 alphas[3] - alphas[4]
   + f_var[0] = 0
 R3: - 2028.416054007841 Fraction[0] + Q0_var[0] = 0
 R4: Q0_var[0] + Lost_0_expr_0 = 985.5016692210908
 R5: Q0_var[0] + Q_hat_adjusted_0 = 1241.399889064605
 R6: Sold_0[0] - Q0_var[0] - Q1_var[0] + total_stock_second_period_0 = 0
 R7: total_stock_second_period_0 + Lost_1_expr_0 = 255.8979409970531
 R8: Sold_1[0] + Left_1[0] - total_stock_second_period_0 = 0
 R9: - 700 Sold_0[0] - 700 Sold_1[0] + 700 Lost_0[0] + 700 Lost_1[0]
   + 300 Left_1[0] + profits_vars[0] = 0
 Fraction_lower_bound_1: Fraction[1] >= 0
 Fraction_upper_bound_1: Fraction[1] <= 1
 R12: - 0.6509235944307136 alphas[0] + 1.28709053145843 alphas[1]
   - 0.856977027260221 alphas[2] - 1.297297962226022 alphas[3] - alphas[4]
   + f_var[1] = 0
 R13: - 2028.416054007841 Fraction[1] + Q0_var[1] = 0
 R14: Q0_var[1] + Lost_0_expr_1 = 1570.720159318349
 R15: Q0_var[1] + Q_hat_adjusted_1 = 2018.892180807597
 R16: Sold_0[1] - Q0_var[1] - Q1_var[1] + total_stock_second_period_1 = 0
 R17: total_stock_second_period_1 + Lost_1_expr_1 = 448.1717444077298
 R18: Sold_1[1] + Left_1[1] - total_stock_second_period_1 = 0
 R19: - 700 Sold_0[1] - 700 Sold_1[1] + 700 Lost_0[1] + 700 Lost_1[1]
   + 300 Left_1[1] + profits_vars[1] = 0
 Fraction_lower_bound_2: Fraction[2] >= 0
 Fraction_upper_bound_2: Fraction[2] <= 1
 R22: - 0.2379962975181053 alphas[0] - 0.6121695908395033 alphas[1]
   + 0.3004769577787669 alphas[2] + 0.3030005211230327 alphas[3]
   - alphas[4] + f_var[2] = 0
 R23: - 2028.416054007841 Fraction[2] + Q0_var[2] = 0
 R24: Q0_var[2] + Lost_0_expr_2 = 1024.019920128306
 R25: Q0_var[2] + Q_hat_adjusted_2 = 1312.896117960501
 R26: Sold_0[2] - Q0_var[2] - Q1_var[2] + total_stock_second_period_2 = 0
 R27: total_stock_second_period_2 + Lost_1_expr_2 = 288.8759196462405
 R28: Sold_1[2] + Left_1[2] - total_stock_second_period_2 = 0
 R29: - 700 Sold_0[2] - 700 Sold_1[2] + 700 Lost_0[2] + 700 Lost_1[2]
   + 300 Left_1[2] + profits_vars[2] = 0
 Fraction_lower_bound_3: Fraction[3] >= 0
 Fraction_upper_bound_3: Fraction[3] <= 1
 R32: - 0.0254092292859822 alphas[0] - 1.538942107976239 alphas[1]
   + 1.020355773715186 alphas[2] + 1.217818518889134 alphas[3] - alphas[4]
   + f_var[3] = 0
 R33: - 2028.416054007841 Fraction[3] + Q0_var[3] = 0
 R34: Q0_var[3] + Lost_0_expr_3 = 841.8614268697045
 R35: Q0_var[3] + Q_hat_adjusted_3 = 1031.663543461739
 R36: Sold_0[3] - Q0_var[3] - Q1_var[3] + total_stock_second_period_3 = 0
 R37: total_stock_second_period_3 + Lost_1_expr_3 = 189.8018387784145
 R38: Sold_1[3] + Left_1[3] - total_stock_second_period_3 = 0
 R39: - 700 Sold_0[3] - 700 Sold_1[3] + 700 Lost_0[3] + 700 Lost_1[3]
   + 300 Left_1[3] + profits_vars[3] = 0
 Fraction_lower_bound_4: Fraction[4] >= 0
 Fraction_upper_bound_4: Fraction[4] <= 1
 R42: 0.4198482119787716 alphas[0] + 1.373869682500199 alphas[1]
   + 1.583203230845045 alphas[2] + 0.0340616211972166 alphas[3] - alphas[4]
   + f_var[4] = 0
 R43: - 2028.416054007841 Fraction[4] + Q0_var[4] = 0
 R44: Q0_var[4] + Lost_0_expr_4 = 944.6580682618797
 R45: Q0_var[4] + Q_hat_adjusted_4 = 1056.997704564391
 R46: Sold_0[4] - Q0_var[4] - Q1_var[4] + total_stock_second_period_4 = 0
 R47: total_stock_second_period_4 + Lost_1_expr_4 = 112.3393641955375
 R48: Sold_1[4] + Left_1[4] - total_stock_second_period_4 = 0
 R49: - 700 Sold_0[4] - 700 Sold_1[4] + 700 Lost_0[4] + 700 Lost_1[4]
   + 300 Left_1[4] + profits_vars[4] = 0
 Fraction_lower_bound_5: Fraction[5] >= 0
 Fraction_upper_bound_5: Fraction[5] <= 1
 R52: - 0.3964103670188668 alphas[0] + 0.0371448923323029 alphas[1]
   + 1.428541733969417 alphas[2] + 0.6692671061127012 alphas[3] - alphas[4]
   + f_var[5] = 0
 R53: - 2028.416054007841 Fraction[5] + Q0_var[5] = 0
 R54: Q0_var[5] + Lost_0_expr_5 = 621.5387210066472
 R55: Q0_var[5] + Q_hat_adjusted_5 = 755.1638181964206
 R56: Sold_0[5] - Q0_var[5] - Q1_var[5] + total_stock_second_period_5 = 0
 R57: total_stock_second_period_5 + Lost_1_expr_5 = 133.6248157314834
 R58: Sold_1[5] + Left_1[5] - total_stock_second_period_5 = 0
 R59: - 700 Sold_0[5] - 700 Sold_1[5] + 700 Lost_0[5] + 700 Lost_1[5]
   + 300 Left_1[5] + profits_vars[5] = 0
 Fraction_lower_bound_6: Fraction[6] >= 0
 Fraction_upper_bound_6: Fraction[6] <= 1
 R62: 0.3686761193000995 alphas[0] - 0.027708689224906 alphas[1]
   - 0.79539451867227 alphas[2] + 1.03130701321808 alphas[3] - alphas[4]
   + f_var[6] = 0
 R63: - 2028.416054007841 Fraction[6] + Q0_var[6] = 0
 R64: Q0_var[6] + Lost_0_expr_6 = 1847.22968764704
 R65: Q0_var[6] + Q_hat_adjusted_6 = 2286.92635238352
 R66: Sold_0[6] - Q0_var[6] - Q1_var[6] + total_stock_second_period_6 = 0
 R67: total_stock_second_period_6 + Lost_1_expr_6 = 439.6963867492836
 R68: Sold_1[6] + Left_1[6] - total_stock_second_period_6 = 0
 R69: - 700 Sold_0[6] - 700 Sold_1[6] + 700 Lost_0[6] + 700 Lost_1[6]
   + 300 Left_1[6] + profits_vars[6] = 0
 Fraction_lower_bound_7: Fraction[7] >= 0
 Fraction_upper_bound_7: Fraction[7] <= 1
 R72: - 1.299494290945913 alphas[0] + 0.0530899639197399 alphas[1]
   + 0.0245662453739224 alphas[2] + 1.318264820481297 alphas[3] - alphas[4]
   + f_var[7] = 0
 R73: - 2028.416054007841 Fraction[7] + Q0_var[7] = 0
 R74: Q0_var[7] + Lost_0_expr_7 = 1042.381783858741
 R75: Q0_var[7] + Q_hat_adjusted_7 = 1369.230483017808
 R76: Sold_0[7] - Q0_var[7] - Q1_var[7] + total_stock_second_period_7 = 0
 R77: total_stock_second_period_7 + Lost_1_expr_7 = 326.8484222780227
 R78: Sold_1[7] + Left_1[7] - total_stock_second_period_7 = 0
 R79: - 700 Sold_0[7] - 700 Sold_1[7] + 700 Lost_0[7] + 700 Lost_1[7]
   + 300 Left_1[7] + profits_vars[7] = 0
 Fraction_lower_bound_8: Fraction[8] >= 0
 Fraction_upper_bound_8: Fraction[8] <= 1
 R82: - 1.563536831972461 alphas[0] + 1.478681245539053 alphas[1]
   - 0.2765481297187581 alphas[2] - 1.121433075671428 alphas[3] - alphas[4]
   + f_var[8] = 0
 R83: - 2028.416054007841 Fraction[8] + Q0_var[8] = 0
 R84: Q0_var[8] + Lost_0_expr_8 = 1257.903530849573
 R85: Q0_var[8] + Q_hat_adjusted_8 = 1626.193413040286
 R86: Sold_0[8] - Q0_var[8] - Q1_var[8] + total_stock_second_period_8 = 0
 R87: total_stock_second_period_8 + Lost_1_expr_8 = 368.2896055543611
 R88: Sold_1[8] + Left_1[8] - total_stock_second_period_8 = 0
 R89: - 700 Sold_0[8] - 700 Sold_1[8] + 700 Lost_0[8] + 700 Lost_1[8]
   + 300 Left_1[8] + profits_vars[8] = 0
 Fraction_lower_bound_9: Fraction[9] >= 0
 Fraction_upper_bound_9: Fraction[9] <= 1
 R92: 0.567546810655553 alphas[0] - 1.5004024101894 alphas[1]
   - 1.5847745563586 alphas[2] + 1.470707008847804 alphas[3] - alphas[4]
   + f_var[9] = 0
 R93: - 2028.416054007841 Fraction[9] + Q0_var[9] = 0
 R94: Q0_var[9] + Lost_0_expr_9 = 2114.35714324075
 R95: Q0_var[9] + Q_hat_adjusted_9 = 2662.693067379602
 R96: Sold_0[9] - Q0_var[9] - Q1_var[9] + total_stock_second_period_9 = 0
 R97: total_stock_second_period_9 + Lost_1_expr_9 = 548.3356465201711
 R98: Sold_1[9] + Left_1[9] - total_stock_second_period_9 = 0
 R99: - 700 Sold_0[9] - 700 Sold_1[9] + 700 Lost_0[9] + 700 Lost_1[9]
   + 300 Left_1[9] + profits_vars[9] = 0
 Fraction_lower_bound_10: Fraction[10] >= 0
 Fraction_upper_bound_10: Fraction[10] <= 1
 R102: - 0.9320300178845095 alphas[0] + 0.5639561514337079 alphas[1]
   - 1.925072238254391 alphas[2] - 1.284715648791757 alphas[3] - alphas[4]
   + f_var[10] = 0
 R103: - 2028.416054007841 Fraction[10] + Q0_var[10] = 0
 R104: Q0_var[10] + Lost_0_expr_10 = 2776.400662008934
 R105: Q0_var[10] + Q_hat_adjusted_10 = 3371.570414250383
 R106: Sold_0[10] - Q0_var[10] - Q1_var[10] + total_stock_second_period_10
   = 0
 R107: total_stock_second_period_10 + Lost_1_expr_10 = 595.1694736983521
 R108: Sold_1[10] + Left_1[10] - total_stock_second_period_10 = 0
 R109: - 700 Sold_0[10] - 700 Sold_1[10] + 700 Lost_0[10] + 700 Lost_1[10]
   + 300 Left_1[10] + profits_vars[10] = 0
 Fraction_lower_bound_11: Fraction[11] >= 0
 Fraction_upper_bound_11: Fraction[11] <= 1
 R112: 0.0333137608874336 alphas[0] - 1.066236411862904 alphas[1]
   + 0.8848679582607427 alphas[2] + 0.9788265905388678 alphas[3]
   - alphas[4] + f_var[11] = 0
 R113: - 2028.416054007841 Fraction[11] + Q0_var[11] = 0
 R114: Q0_var[11] + Lost_0_expr_11 = 1085.780691702743
 R115: Q0_var[11] + Q_hat_adjusted_11 = 1294.229461207121
 R116: Sold_0[11] - Q0_var[11] - Q1_var[11] + total_stock_second_period_11
   = 0
 R117: total_stock_second_period_11 + Lost_1_expr_11 = 208.4484921991337
 R118: Sold_1[11] + Left_1[11] - total_stock_second_period_11 = 0
 R119: - 700 Sold_0[11] - 700 Sold_1[11] + 700 Lost_0[11] + 700 Lost_1[11]
   + 300 Left_1[11] + profits_vars[11] = 0
 Fraction_lower_bound_12: Fraction[12] >= 0
 Fraction_upper_bound_12: Fraction[12] <= 1
 R122: - 0.1104782175581155 alphas[0] - 0.4663535360785392 alphas[1]
   - 0.7361231455833085 alphas[2] + 1.223550772776169 alphas[3] - alphas[4]
   + f_var[12] = 0
 R123: - 2028.416054007841 Fraction[12] + Q0_var[12] = 0
 R124: Q0_var[12] + Lost_0_expr_12 = 1803.127894967017
 R125: Q0_var[12] + Q_hat_adjusted_12 = 2234.667275303906
 R126: Sold_0[12] - Q0_var[12] - Q1_var[12] + total_stock_second_period_12
   = 0
 R127: total_stock_second_period_12 + Lost_1_expr_12 = 431.5391015500504
 R128: Sold_1[12] + Left_1[12] - total_stock_second_period_12 = 0
 R129: - 700 Sold_0[12] - 700 Sold_1[12] + 700 Lost_0[12] + 700 Lost_1[12]
   + 300 Left_1[12] + profits_vars[12] = 0
 Fraction_lower_bound_13: Fraction[13] >= 0
 Fraction_upper_bound_13: Fraction[13] <= 1
 R132: - 1.423724490489904 alphas[0] + 1.175103588620279 alphas[1]
   + 0.7162029745743949 alphas[2] + 1.186960952297952 alphas[3] - alphas[4]
   + f_var[13] = 0
 R133: - 2028.416054007841 Fraction[13] + Q0_var[13] = 0
 R134: Q0_var[13] + Lost_0_expr_13 = 1026.645354231385
 R135: Q0_var[13] + Q_hat_adjusted_13 = 1258.306820757196
 R136: Sold_0[13] - Q0_var[13] - Q1_var[13] + total_stock_second_period_13
   = 0
 R137: total_stock_second_period_13 + Lost_1_expr_13 = 231.6611883685753
 R138: Sold_1[13] + Left_1[13] - total_stock_second_period_13 = 0
 R139: - 700 Sold_0[13] - 700 Sold_1[13] + 700 Lost_0[13] + 700 Lost_1[13]
   + 300 Left_1[13] + profits_vars[13] = 0
 Fraction_lower_bound_14: Fraction[14] >= 0
 Fraction_upper_bound_14: Fraction[14] <= 1
 R142: 1.714974928057006 alphas[0] - 0.9330929258103161 alphas[1]
   + 1.597836103058035 alphas[2] - 0.9270805016014423 alphas[3] - alphas[4]
   + f_var[14] = 0
 R143: - 2028.416054007841 Fraction[14] + Q0_var[14] = 0
 R144: Q0_var[14] + Lost_0_expr_14 = 487.4053591546971
 R145: Q0_var[14] + Q_hat_adjusted_14 = 597.7311343591938
 R146: Sold_0[14] - Q0_var[14] - Q1_var[14] + total_stock_second_period_14
   = 0
 R147: total_stock_second_period_14 + Lost_1_expr_14 = 110.32549913407
 R148: Sold_1[14] + Left_1[14] - total_stock_second_period_14 = 0
 R149: - 700 Sold_0[14] - 700 Sold_1[14] + 700 Lost_0[14] + 700 Lost_1[14]
   + 300 Left_1[14] + profits_vars[14] = 0
 Fraction_lower_bound_15: Fraction[15] >= 0
 Fraction_upper_bound_15: Fraction[15] <= 1
 R152: 1.655866365215913 alphas[0] + 0.7706258476638413 alphas[1]
   - 1.08235283065384 alphas[2] + 0.6215993257875255 alphas[3] - alphas[4]
   + f_var[15] = 0
 R153: - 2028.416054007841 Fraction[15] + Q0_var[15] = 0
 R154: Q0_var[15] + Lost_0_expr_15 = 1929.829545184692
 R155: Q0_var[15] + Q_hat_adjusted_15 = 2409.019149629327
 R156: Sold_0[15] - Q0_var[15] - Q1_var[15] + total_stock_second_period_15
   = 0
 R157: total_stock_second_period_15 + Lost_1_expr_15 = 479.1893269180707
 R158: Sold_1[15] + Left_1[15] - total_stock_second_period_15 = 0
 R159: - 700 Sold_0[15] - 700 Sold_1[15] + 700 Lost_0[15] + 700 Lost_1[15]
   + 300 Left_1[15] + profits_vars[15] = 0
 Fraction_lower_bound_16: Fraction[16] >= 0
 Fraction_upper_bound_16: Fraction[16] <= 1
 R162: 1.901622154698307 alphas[0] - 1.207887106810452 alphas[1]
   + 0.5102289878293261 alphas[2] - 0.0142058691414214 alphas[3]
   - alphas[4] + f_var[16] = 0
 R163: - 2028.416054007841 Fraction[16] + Q0_var[16] = 0
 R164: Q0_var[16] + Lost_0_expr_16 = 877.6680076190777
 R165: Q0_var[16] + Q_hat_adjusted_16 = 1137.676860756748
 R166: Sold_0[16] - Q0_var[16] - Q1_var[16] + total_stock_second_period_16
   = 0
 R167: total_stock_second_period_16 + Lost_1_expr_16 = 260.0085754112481
 R168: Sold_1[16] + Left_1[16] - total_stock_second_period_16 = 0
 R169: - 700 Sold_0[16] - 700 Sold_1[16] + 700 Lost_0[16] + 700 Lost_1[16]
   + 300 Left_1[16] + profits_vars[16] = 0
 Fraction_lower_bound_17: Fraction[17] >= 0
 Fraction_upper_bound_17: Fraction[17] <= 1
 R172: - 1.082231784035022 alphas[0] + 0.0162591891519979 alphas[1]
   + 0.2797248737463011 alphas[2] - 0.5108415636449818 alphas[3]
   - alphas[4] + f_var[17] = 0
 R173: - 2028.416054007841 Fraction[17] + Q0_var[17] = 0
 R174: Q0_var[17] + Lost_0_expr_17 = 1000.389772420226
 R175: Q0_var[17] + Q_hat_adjusted_17 = 1292.121996752742
 R176: Sold_0[17] - Q0_var[17] - Q1_var[17] + total_stock_second_period_17
   = 0
 R177: total_stock_second_period_17 + Lost_1_expr_17 = 291.731947085949
 R178: Sold_1[17] + Left_1[17] - total_stock_second_period_17 = 0
 R179: - 700 Sold_0[17] - 700 Sold_1[17] + 700 Lost_0[17] + 700 Lost_1[17]
   + 300 Left_1[17] + profits_vars[17] = 0
 Fraction_lower_bound_18: Fraction[18] >= 0
 Fraction_upper_bound_18: Fraction[18] <= 1
 R182: - 0.8821953122750865 alphas[0] - 1.210468107017091 alphas[1]
   - 0.4645136380804882 alphas[2] - 1.611272289942189 alphas[3] - alphas[4]
   + f_var[18] = 0
 R183: - 2028.416054007841 Fraction[18] + Q0_var[18] = 0
 R184: Q0_var[18] + Lost_0_expr_18 = 1663.667313796551
 R185: Q0_var[18] + Q_hat_adjusted_18 = 2057.826148612096
 R186: Sold_0[18] - Q0_var[18] - Q1_var[18] + total_stock_second_period_18
   = 0
 R187: total_stock_second_period_18 + Lost_1_expr_18 = 394.1585567768029
 R188: Sold_1[18] + Left_1[18] - total_stock_second_period_18 = 0
 R189: - 700 Sold_0[18] - 700 Sold_1[18] + 700 Lost_0[18] + 700 Lost_1[18]
   + 300 Left_1[18] + profits_vars[18] = 0
 Fraction_lower_bound_19: Fraction[19] >= 0
 Fraction_upper_bound_19: Fraction[19] <= 1
 R192: - 1.219569266564456 alphas[0] - 0.4379875488601938 alphas[1]
   - 1.346364619250364 alphas[2] - 1.126695627034258 alphas[3] - alphas[4]
   + f_var[19] = 0
 R193: - 2028.416054007841 Fraction[19] + Q0_var[19] = 0
 R194: Q0_var[19] + Lost_0_expr_19 = 1980.356078446152
 R195: Q0_var[19] + Q_hat_adjusted_19 = 2495.880583074765
 R196: Sold_0[19] - Q0_var[19] - Q1_var[19] + total_stock_second_period_19
   = 0
 R197: total_stock_second_period_19 + Lost_1_expr_19 = 515.5242276180952
 R198: Sold_1[19] + Left_1[19] - total_stock_second_period_19 = 0
 R199: - 700 Sold_0[19] - 700 Sold_1[19] + 700 Lost_0[19] + 700 Lost_1[19]
   + 300 Left_1[19] + profits_vars[19] = 0
 Fraction_lower_bound_20: Fraction[20] >= 0
 Fraction_upper_bound_20: Fraction[20] <= 1
 R202: - 1.618466905718923 alphas[0] - 0.5715876425516639 alphas[1]
   - 0.6122499453658867 alphas[2] + 0.5310019898690563 alphas[3]
   - alphas[4] + f_var[20] = 0
 R203: - 2028.416054007841 Fraction[20] + Q0_var[20] = 0
 R204: Q0_var[20] + Lost_0_expr_20 = 1475.395085876261
 R205: Q0_var[20] + Q_hat_adjusted_20 = 1889.886285788248
 R206: Sold_0[20] - Q0_var[20] - Q1_var[20] + total_stock_second_period_20
   = 0
 R207: total_stock_second_period_20 + Lost_1_expr_20 = 414.4909217955744
 R208: Sold_1[20] + Left_1[20] - total_stock_second_period_20 = 0
 R209: - 700 Sold_0[20] - 700 Sold_1[20] + 700 Lost_0[20] + 700 Lost_1[20]
   + 300 Left_1[20] + profits_vars[20] = 0
 Fraction_lower_bound_21: Fraction[21] >= 0
 Fraction_upper_bound_21: Fraction[21] <= 1
 R212: - 0.959332475660104 alphas[0] + 0.3426726329578249 alphas[1]
   - 1.497506786639079 alphas[2] + 1.034954163865374 alphas[3] - alphas[4]
   + f_var[21] = 0
 R213: - 2028.416054007841 Fraction[21] + Q0_var[21] = 0
 R214: Q0_var[21] + Lost_0_expr_21 = 1997.067163955949
 R215: Q0_var[21] + Q_hat_adjusted_21 = 2533.392768903952
 R216: Sold_0[21] - Q0_var[21] - Q1_var[21] + total_stock_second_period_21
   = 0
 R217: total_stock_second_period_21 + Lost_1_expr_21 = 536.3253277236977
 R218: Sold_1[21] + Left_1[21] - total_stock_second_period_21 = 0
 R219: - 700 Sold_0[21] - 700 Sold_1[21] + 700 Lost_0[21] + 700 Lost_1[21]
   + 300 Left_1[21] + profits_vars[21] = 0
 Fraction_lower_bound_22: Fraction[22] >= 0
 Fraction_upper_bound_22: Fraction[22] <= 1
 R222: 0.2809230890672019 alphas[0] - 1.514016999219016 alphas[1]
   + 0.6790746192443368 alphas[2] + 1.31119748641023 alphas[3] - alphas[4]
   + f_var[22] = 0
 R223: - 2028.416054007841 Fraction[22] + Q0_var[22] = 0
 R224: Q0_var[22] + Lost_0_expr_22 = 1091.077127926938
 R225: Q0_var[22] + Q_hat_adjusted_22 = 1327.848424262717
 R226: Sold_0[22] - Q0_var[22] - Q1_var[22] + total_stock_second_period_22
   = 0
 R227: total_stock_second_period_22 + Lost_1_expr_22 = 236.7710174077933
 R228: Sold_1[22] + Left_1[22] - total_stock_second_period_22 = 0
 R229: - 700 Sold_0[22] - 700 Sold_1[22] + 700 Lost_0[22] + 700 Lost_1[22]
   + 300 Left_1[22] + profits_vars[22] = 0
 Fraction_lower_bound_23: Fraction[23] >= 0
 Fraction_upper_bound_23: Fraction[23] <= 1
 R232: - 0.890908948497018 alphas[0] - 0.2398029125610173 alphas[1]
   - 1.226468639016819 alphas[2] + 0.4175017560904177 alphas[3] - alphas[4]
   + f_var[23] = 0
 R233: - 2028.416054007841 Fraction[23] + Q0_var[23] = 0
 R234: Q0_var[23] + Lost_0_expr_23 = 2145.597831988234
 R235: Q0_var[23] + Q_hat_adjusted_23 = 2644.621527966629
 R236: Sold_0[23] - Q0_var[23] - Q1_var[23] + total_stock_second_period_23
   = 0
 R237: total_stock_second_period_23 + Lost_1_expr_23 = 499.0234169562818
 R238: Sold_1[23] + Left_1[23] - total_stock_second_period_23 = 0
 R239: - 700 Sold_0[23] - 700 Sold_1[23] + 700 Lost_0[23] + 700 Lost_1[23]
   + 300 Left_1[23] + profits_vars[23] = 0
 Fraction_lower_bound_24: Fraction[24] >= 0
 Fraction_upper_bound_24: Fraction[24] <= 1
 R242: 1.54147401940782 alphas[0] + 0.6589641810516157 alphas[1]
   + 0.9984403493944379 alphas[2] + 0.5608001189097406 alphas[3]
   - alphas[4] + f_var[24] = 0
 R243: - 2028.416054007841 Fraction[24] + Q0_var[24] = 0
 R244: Q0_var[24] + Lost_0_expr_24 = 803.8390560387079
 R245: Q0_var[24] + Q_hat_adjusted_24 = 996.6573066940407
 R246: Sold_0[24] - Q0_var[24] - Q1_var[24] + total_stock_second_period_24
   = 0
 R247: total_stock_second_period_24 + Lost_1_expr_24 = 192.8179721529758
 R248: Sold_1[24] + Left_1[24] - total_stock_second_period_24 = 0
 R249: - 700 Sold_0[24] - 700 Sold_1[24] + 700 Lost_0[24] + 700 Lost_1[24]
   + 300 Left_1[24] + profits_vars[24] = 0
 Fraction_lower_bound_25: Fraction[25] >= 0
 Fraction_upper_bound_25: Fraction[25] <= 1
 R252: - 0.3744719111560164 alphas[0] - 0.0863020612640085 alphas[1]
   - 1.788661692043826 alphas[2] - 0.1434236163560723 alphas[3] - alphas[4]
   + f_var[25] = 0
 R253: - 2028.416054007841 Fraction[25] + Q0_var[25] = 0
 R254: Q0_var[25] + Lost_0_expr_25 = 2289.88115278488
 R255: Q0_var[25] + Q_hat_adjusted_25 = 2866.277258732909
 R256: Sold_0[25] - Q0_var[25] - Q1_var[25] + total_stock_second_period_25
   = 0
 R257: total_stock_second_period_25 + Lost_1_expr_25 = 576.3958284859723
 R258: Sold_1[25] + Left_1[25] - total_stock_second_period_25 = 0
 R259: - 700 Sold_0[25] - 700 Sold_1[25] + 700 Lost_0[25] + 700 Lost_1[25]
   + 300 Left_1[25] + profits_vars[25] = 0
 Fraction_lower_bound_26: Fraction[26] >= 0
 Fraction_upper_bound_26: Fraction[26] <= 1
 R262: 1.449362342404136 alphas[0] + 2.311116297307088 alphas[1]
   - 0.824805748044052 alphas[2] - 0.9678537137090004 alphas[3] - alphas[4]
   + f_var[26] = 0
 R263: - 2028.416054007841 Fraction[26] + Q0_var[26] = 0
 R264: Q0_var[26] + Lost_0_expr_26 = 1936.480774365079
 R265: Q0_var[26] + Q_hat_adjusted_26 = 2380.225190676716
 R266: Sold_0[26] - Q0_var[26] - Q1_var[26] + total_stock_second_period_26
   = 0
 R267: total_stock_second_period_26 + Lost_1_expr_26 = 443.7441381939076
 R268: Sold_1[26] + Left_1[26] - total_stock_second_period_26 = 0
 R269: - 700 Sold_0[26] - 700 Sold_1[26] + 700 Lost_0[26] + 700 Lost_1[26]
   + 300 Left_1[26] + profits_vars[26] = 0
 Fraction_lower_bound_27: Fraction[27] >= 0
 Fraction_upper_bound_27: Fraction[27] <= 1
 R272: - 1.493774703007406 alphas[0] + 1.158209523202141 alphas[1]
   + 0.8899153575123454 alphas[2] - 1.012657893691021 alphas[3] - alphas[4]
   + f_var[27] = 0
 R273: - 2028.416054007841 Fraction[27] + Q0_var[27] = 0
 R274: Q0_var[27] + Lost_0_expr_27 = 888.6520263497487
 R275: Q0_var[27] + Q_hat_adjusted_27 = 1096.406143860195
 R276: Sold_0[27] - Q0_var[27] - Q1_var[27] + total_stock_second_period_27
   = 0
 R277: total_stock_second_period_27 + Lost_1_expr_27 = 207.7538385567793
 R278: Sold_1[27] + Left_1[27] - total_stock_second_period_27 = 0
 R279: - 700 Sold_0[27] - 700 Sold_1[27] + 700 Lost_0[27] + 700 Lost_1[27]
   + 300 Left_1[27] + profits_vars[27] = 0
 Fraction_lower_bound_28: Fraction[28] >= 0
 Fraction_upper_bound_28: Fraction[28] <= 1
 R282: 0.0591950776359468 alphas[0] - 0.3058761945937323 alphas[1]
   - 1.768966778466122 alphas[2] + 0.4831241513113188 alphas[3] - alphas[4]
   + f_var[28] = 0
 R283: - 2028.416054007841 Fraction[28] + Q0_var[28] = 0
 R284: Q0_var[28] + Lost_0_expr_28 = 2420.221455590129
 R285: Q0_var[28] + Q_hat_adjusted_28 = 2993.907028851979
 R286: Sold_0[28] - Q0_var[28] - Q1_var[28] + total_stock_second_period_28
   = 0
 R287: total_stock_second_period_28 + Lost_1_expr_28 = 573.6852952444622
 R288: Sold_1[28] + Left_1[28] - total_stock_second_period_28 = 0
 R289: - 700 Sold_0[28] - 700 Sold_1[28] + 700 Lost_0[28] + 700 Lost_1[28]
   + 300 Left_1[28] + profits_vars[28] = 0
 Fraction_lower_bound_29: Fraction[29] >= 0
 Fraction_upper_bound_29: Fraction[29] <= 1
 R292: 0.4528779152734247 alphas[0] + 1.205150293466185 alphas[1]
   - 0.9823614182491538 alphas[2] + 0.1332012933893546 alphas[3]
   - alphas[4] + f_var[29] = 0
 R293: - 2028.416054007841 Fraction[29] + Q0_var[29] = 0
 R294: Q0_var[29] + Lost_0_expr_29 = 1415.696311696586
 R295: Q0_var[29] + Q_hat_adjusted_29 = 1881.124491514184
 R296: Sold_0[29] - Q0_var[29] - Q1_var[29] + total_stock_second_period_29
   = 0
 R297: total_stock_second_period_29 + Lost_1_expr_29 = 465.4279033850789
 R298: Sold_1[29] + Left_1[29] - total_stock_second_period_29 = 0
 R299: - 700 Sold_0[29] - 700 Sold_1[29] + 700 Lost_0[29] + 700 Lost_1[29]
   + 300 Left_1[29] + profits_vars[29] = 0
 Fraction_lower_bound_30: Fraction[30] >= 0
 Fraction_upper_bound_30: Fraction[30] <= 1
 R302: 1.004200612610371 alphas[0] - 0.1337532032057398 alphas[1]
   + 0.750283491536 alphas[2] + 0.854678609519227 alphas[3] - alphas[4]
   + f_var[30] = 0
 R303: - 2028.416054007841 Fraction[30] + Q0_var[30] = 0
 R304: Q0_var[30] + Lost_0_expr_30 = 869.7617671527879
 R305: Q0_var[30] + Q_hat_adjusted_30 = 1096.732866974807
 R306: Sold_0[30] - Q0_var[30] - Q1_var[30] + total_stock_second_period_30
   = 0
 R307: total_stock_second_period_30 + Lost_1_expr_30 = 226.9708212975124
 R308: Sold_1[30] + Left_1[30] - total_stock_second_period_30 = 0
 R309: - 700 Sold_0[30] - 700 Sold_1[30] + 700 Lost_0[30] + 700 Lost_1[30]
   + 300 Left_1[30] + profits_vars[30] = 0
 Fraction_lower_bound_31: Fraction[31] >= 0
 Fraction_upper_bound_31: Fraction[31] <= 1
 R312: - 0.8677863737108036 alphas[0] + 0.6389027366982564 alphas[1]
   + 0.8978924072085647 alphas[2] - 1.349920552576007 alphas[3] - alphas[4]
   + f_var[31] = 0
 R313: - 2028.416054007841 Fraction[31] + Q0_var[31] = 0
 R314: Q0_var[31] + Lost_0_expr_31 = 953.6834708503509
 R315: Q0_var[31] + Q_hat_adjusted_31 = 1160.339737466994
 R316: Sold_0[31] - Q0_var[31] - Q1_var[31] + total_stock_second_period_31
   = 0
 R317: total_stock_second_period_31 + Lost_1_expr_31 = 206.6559886837406
 R318: Sold_1[31] + Left_1[31] - total_stock_second_period_31 = 0
 R319: - 700 Sold_0[31] - 700 Sold_1[31] + 700 Lost_0[31] + 700 Lost_1[31]
   + 300 Left_1[31] + profits_vars[31] = 0
 Fraction_lower_bound_32: Fraction[32] >= 0
 Fraction_upper_bound_32: Fraction[32] <= 1
 R322: 0.3004959828634036 alphas[0] + 1.8363651474761 alphas[1]
   - 0.2097784424729602 alphas[2] + 1.305848341828286 alphas[3] - alphas[4]
   + f_var[32] = 0
 R323: - 2028.416054007841 Fraction[32] + Q0_var[32] = 0
 R324: Q0_var[32] + Lost_0_expr_32 = 1410.101183009175
 R325: Q0_var[32] + Q_hat_adjusted_32 = 1769.20181781339
 R326: Sold_0[32] - Q0_var[32] - Q1_var[32] + total_stock_second_period_32
   = 0
 R327: total_stock_second_period_32 + Lost_1_expr_32 = 359.1003569653316
 R328: Sold_1[32] + Left_1[32] - total_stock_second_period_32 = 0
 R329: - 700 Sold_0[32] - 700 Sold_1[32] + 700 Lost_0[32] + 700 Lost_1[32]
   + 300 Left_1[32] + profits_vars[32] = 0
 Fraction_lower_bound_33: Fraction[33] >= 0
 Fraction_upper_bound_33: Fraction[33] <= 1
 R332: - 0.1119083925633281 alphas[0] + 1.171652093844782 alphas[1]
   + 1.579628363899807 alphas[2] - 1.017636964542434 alphas[3] - alphas[4]
   + f_var[33] = 0
 R333: - 2028.416054007841 Fraction[33] + Q0_var[33] = 0
 R334: Q0_var[33] + Lost_0_expr_33 = 553.121104136178
 R335: Q0_var[33] + Q_hat_adjusted_33 = 665.9527389674838
 R336: Sold_0[33] - Q0_var[33] - Q1_var[33] + total_stock_second_period_33
   = 0
 R337: total_stock_second_period_33 + Lost_1_expr_33 = 112.8313590284054
 R338: Sold_1[33] + Left_1[33] - total_stock_second_period_33 = 0
 R339: - 700 Sold_0[33] - 700 Sold_1[33] + 700 Lost_0[33] + 700 Lost_1[33]
   + 300 Left_1[33] + profits_vars[33] = 0
 Fraction_lower_bound_34: Fraction[34] >= 0
 Fraction_upper_bound_34: Fraction[34] <= 1
 R342: 1.906869221658971 alphas[0] + 0.0624273002039462 alphas[1]
   + 0.9191162298511577 alphas[2] - 1.314168571815939 alphas[3] - alphas[4]
   + f_var[34] = 0
 R343: - 2028.416054007841 Fraction[34] + Q0_var[34] = 0
 R344: Q0_var[34] + Lost_0_expr_34 = 941.8925750063599
 R345: Q0_var[34] + Q_hat_adjusted_34 = 1145.627890748708
 R346: Sold_0[34] - Q0_var[34] - Q1_var[34] + total_stock_second_period_34
   = 0
 R347: total_stock_second_period_34 + Lost_1_expr_34 = 203.7350377205705
 R348: Sold_1[34] + Left_1[34] - total_stock_second_period_34 = 0
 R349: - 700 Sold_0[34] - 700 Sold_1[34] + 700 Lost_0[34] + 700 Lost_1[34]
   + 300 Left_1[34] + profits_vars[34] = 0
 Fraction_lower_bound_35: Fraction[35] >= 0
 Fraction_upper_bound_35: Fraction[35] <= 1
 R352: - 0.2926198299344603 alphas[0] - 0.022935205567689 alphas[1]
   + 0.1299655318483328 alphas[2] - 1.309494249403513 alphas[3] - alphas[4]
   + f_var[35] = 0
 R353: - 2028.416054007841 Fraction[35] + Q0_var[35] = 0
 R354: Q0_var[35] + Lost_0_expr_35 = 1059.166411412334
 R355: Q0_var[35] + Q_hat_adjusted_35 = 1371.509424346211
 R356: Sold_0[35] - Q0_var[35] - Q1_var[35] + total_stock_second_period_35
   = 0
 R357: total_stock_second_period_35 + Lost_1_expr_35 = 312.3427343757531
 R358: Sold_1[35] + Left_1[35] - total_stock_second_period_35 = 0
 R359: - 700 Sold_0[35] - 700 Sold_1[35] + 700 Lost_0[35] + 700 Lost_1[35]
   + 300 Left_1[35] + profits_vars[35] = 0
 Fraction_lower_bound_36: Fraction[36] >= 0
 Fraction_upper_bound_36: Fraction[36] <= 1
 R362: - 0.272272897315425 alphas[0] + 0.0446249254683583 alphas[1]
   + 0.3134899696743634 alphas[2] + 0.570584013404363 alphas[3] - alphas[4]
   + f_var[36] = 0
 R363: - 2028.416054007841 Fraction[36] + Q0_var[36] = 0
 R364: Q0_var[36] + Lost_0_expr_36 = 999.2920081744734
 R365: Q0_var[36] + Q_hat_adjusted_36 = 1286.377276368151
 R366: Sold_0[36] - Q0_var[36] - Q1_var[36] + total_stock_second_period_36
   = 0
 R367: total_stock_second_period_36 + Lost_1_expr_36 = 287.0849901671128
 R368: Sold_1[36] + Left_1[36] - total_stock_second_period_36 = 0
 R369: - 700 Sold_0[36] - 700 Sold_1[36] + 700 Lost_0[36] + 700 Lost_1[36]
   + 300 Left_1[36] + profits_vars[36] = 0
 Fraction_lower_bound_37: Fraction[37] >= 0
 Fraction_upper_bound_37: Fraction[37] <= 1
 R372: - 0.290043303091715 alphas[0] - 0.2774080051893102 alphas[1]
   - 0.0113229613372425 alphas[2] + 0.1553920060914088 alphas[3]
   - alphas[4] + f_var[37] = 0
 R373: - 2028.416054007841 Fraction[37] + Q0_var[37] = 0
 R374: Q0_var[37] + Lost_0_expr_37 = 1218.506327701144
 R375: Q0_var[37] + Q_hat_adjusted_37 = 1550.294317752225
 R376: Sold_0[37] - Q0_var[37] - Q1_var[37] + total_stock_second_period_37
   = 0
 R377: total_stock_second_period_37 + Lost_1_expr_37 = 331.7877121824054
 R378: Sold_1[37] + Left_1[37] - total_stock_second_period_37 = 0
 R379: - 700 Sold_0[37] - 700 Sold_1[37] + 700 Lost_0[37] + 700 Lost_1[37]
   + 300 Left_1[37] + profits_vars[37] = 0
 Fraction_lower_bound_38: Fraction[38] >= 0
 Fraction_upper_bound_38: Fraction[38] <= 1
 R382: - 1.490392572539918 alphas[0] - 0.2729269321273108 alphas[1]
   + 0.6642276067877726 alphas[2] - 0.6205674296544516 alphas[3]
   - alphas[4] + f_var[38] = 0
 R383: - 2028.416054007841 Fraction[38] + Q0_var[38] = 0
 R384: Q0_var[38] + Lost_0_expr_38 = 1324.417089681451
 R385: Q0_var[38] + Q_hat_adjusted_38 = 1563.231721339206
 R386: Sold_0[38] - Q0_var[38] - Q1_var[38] + total_stock_second_period_38
   = 0
 R387: total_stock_second_period_38 + Lost_1_expr_38 = 238.8143531473638
 R388: Sold_1[38] + Left_1[38] - total_stock_second_period_38 = 0
 R389: - 700 Sold_0[38] - 700 Sold_1[38] + 700 Lost_0[38] + 700 Lost_1[38]
   + 300 Left_1[38] + profits_vars[38] = 0
 Fraction_lower_bound_39: Fraction[39] >= 0
 Fraction_upper_bound_39: Fraction[39] <= 1
 R392: - 0.5283629773439216 alphas[0] + 0.6280844091501678 alphas[1]
   - 0.4589450445991258 alphas[2] + 0.5219497780887212 alphas[3]
   - alphas[4] + f_var[39] = 0
 R393: - 2028.416054007841 Fraction[39] + Q0_var[39] = 0
 R394: Q0_var[39] + Lost_0_expr_39 = 1600.632086988877
 R395: Q0_var[39] + Q_hat_adjusted_39 = 1994.024537244877
 R396: Sold_0[39] - Q0_var[39] - Q1_var[39] + total_stock_second_period_39
   = 0
 R397: total_stock_second_period_39 + Lost_1_expr_39 = 393.3921732290844
 R398: Sold_1[39] + Left_1[39] - total_stock_second_period_39 = 0
 R399: - 700 Sold_0[39] - 700 Sold_1[39] + 700 Lost_0[39] + 700 Lost_1[39]
   + 300 Left_1[39] + profits_vars[39] = 0
 Fraction_lower_bound_40: Fraction[40] >= 0
 Fraction_upper_bound_40: Fraction[40] <= 1
 R402: 0.6554521452111429 alphas[0] - 1.270939366109456 alphas[1]
   - 1.465554081238291 alphas[2] + 0.1661368510093704 alphas[3] - alphas[4]
   + f_var[40] = 0
 R403: - 2028.416054007841 Fraction[40] + Q0_var[40] = 0
 R404: Q0_var[40] + Lost_0_expr_40 = 2091.203152144836
 R405: Q0_var[40] + Q_hat_adjusted_40 = 2623.131233296414
 R406: Sold_0[40] - Q0_var[40] - Q1_var[40] + total_stock_second_period_40
   = 0
 R407: total_stock_second_period_40 + Lost_1_expr_40 = 531.9278029616157
 R408: Sold_1[40] + Left_1[40] - total_stock_second_period_40 = 0
 R409: - 700 Sold_0[40] - 700 Sold_1[40] + 700 Lost_0[40] + 700 Lost_1[40]
   + 300 Left_1[40] + profits_vars[40] = 0
 Fraction_lower_bound_41: Fraction[41] >= 0
 Fraction_upper_bound_41: Fraction[41] <= 1
 R412: 0.3707155139526951 alphas[0] + 0.88879779138409 alphas[1]
   + 1.245864751663434 alphas[2] + 0.7969559274606892 alphas[3] - alphas[4]
   + f_var[41] = 0
 R413: - 2028.416054007841 Fraction[41] + Q0_var[41] = 0
 R414: Q0_var[41] + Lost_0_expr_41 = 640.3005463220409
 R415: Q0_var[41] + Q_hat_adjusted_41 = 799.0667530833817
 R416: Sold_0[41] - Q0_var[41] - Q1_var[41] + total_stock_second_period_41
   = 0
 R417: total_stock_second_period_41 + Lost_1_expr_41 = 158.7659279810111
 R418: Sold_1[41] + Left_1[41] - total_stock_second_period_41 = 0
 R419: - 700 Sold_0[41] - 700 Sold_1[41] + 700 Lost_0[41] + 700 Lost_1[41]
   + 300 Left_1[41] + profits_vars[41] = 0
 Fraction_lower_bound_42: Fraction[42] >= 0
 Fraction_upper_bound_42: Fraction[42] <= 1
 R422: - 0.5864345217416338 alphas[0] + 0.609930294595618 alphas[1]
   - 0.206793517918046 alphas[2] - 0.1569087301744184 alphas[3] - alphas[4]
   + f_var[42] = 0
 R423: - 2028.416054007841 Fraction[42] + Q0_var[42] = 0
 R424: Q0_var[42] + Lost_0_expr_42 = 1809.380251590136
 R425: Q0_var[42] + Q_hat_adjusted_42 = 2168.07008365509
 R426: Sold_0[42] - Q0_var[42] - Q1_var[42] + total_stock_second_period_42
   = 0
 R427: total_stock_second_period_42 + Lost_1_expr_42 = 358.6895535770571
 R428: Sold_1[42] + Left_1[42] - total_stock_second_period_42 = 0
 R429: - 700 Sold_0[42] - 700 Sold_1[42] + 700 Lost_0[42] + 700 Lost_1[42]
   + 300 Left_1[42] + profits_vars[42] = 0
 Fraction_lower_bound_43: Fraction[43] >= 0
 Fraction_upper_bound_43: Fraction[43] <= 1
 R432: 1.754680927335964 alphas[0] - 1.252063613118527 alphas[1]
   + 1.193054073923892 alphas[2] + 1.350259309431124 alphas[3] - alphas[4]
   + f_var[43] = 0
 R433: - 2028.416054007841 Fraction[43] + Q0_var[43] = 0
 R434: Q0_var[43] + Lost_0_expr_43 = 657.0729437534862
 R435: Q0_var[43] + Q_hat_adjusted_43 = 823.1072762094914
 R436: Sold_0[43] - Q0_var[43] - Q1_var[43] + total_stock_second_period_43
   = 0
 R437: total_stock_second_period_43 + Lost_1_expr_43 = 166.034053171934
 R438: Sold_1[43] + Left_1[43] - total_stock_second_period_43 = 0
 R439: - 700 Sold_0[43] - 700 Sold_1[43] + 700 Lost_0[43] + 700 Lost_1[43]
   + 300 Left_1[43] + profits_vars[43] = 0
 Fraction_lower_bound_44: Fraction[44] >= 0
 Fraction_upper_bound_44: Fraction[44] <= 1
 R442: - 0.4730729541736584 alphas[0] - 0.9020480301699687 alphas[1]
   - 0.9315130317672309 alphas[2] - 1.390754901918511 alphas[3] - alphas[4]
   + f_var[44] = 0
 R443: - 2028.416054007841 Fraction[44] + Q0_var[44] = 0
 R444: Q0_var[44] + Lost_0_expr_44 = 1895.918504047045
 R445: Q0_var[44] + Q_hat_adjusted_44 = 2354.348623362157
 R446: Sold_0[44] - Q0_var[44] - Q1_var[44] + total_stock_second_period_44
   = 0
 R447: total_stock_second_period_44 + Lost_1_expr_44 = 458.4298405962969
 R448: Sold_1[44] + Left_1[44] - total_stock_second_period_44 = 0
 R449: - 700 Sold_0[44] - 700 Sold_1[44] + 700 Lost_0[44] + 700 Lost_1[44]
   + 300 Left_1[44] + profits_vars[44] = 0
 Fraction_lower_bound_45: Fraction[45] >= 0
 Fraction_upper_bound_45: Fraction[45] <= 1
 R452: - 0.4872912432374461 alphas[0] - 0.4843558456788744 alphas[1]
   + 0.2339638849644033 alphas[2] + 1.232863003333669 alphas[3] - alphas[4]
   + f_var[45] = 0
 R453: - 2028.416054007841 Fraction[45] + Q0_var[45] = 0
 R454: Q0_var[45] + Lost_0_expr_45 = 1052.136144779188
 R455: Q0_var[45] + Q_hat_adjusted_45 = 1350.166274010808
 R456: Sold_0[45] - Q0_var[45] - Q1_var[45] + total_stock_second_period_45
   = 0
 R457: total_stock_second_period_45 + Lost_1_expr_45 = 298.0298514036469
 R458: Sold_1[45] + Left_1[45] - total_stock_second_period_45 = 0
 R459: - 700 Sold_0[45] - 700 Sold_1[45] + 700 Lost_0[45] + 700 Lost_1[45]
   + 300 Left_1[45] + profits_vars[45] = 0
 Fraction_lower_bound_46: Fraction[46] >= 0
 Fraction_upper_bound_46: Fraction[46] <= 1
 R462: 1.20317178882866 alphas[0] + 1.980272209355848 alphas[1]
   - 0.3813324343285022 alphas[2] - 0.3709798611131848 alphas[3]
   - alphas[4] + f_var[46] = 0
 R463: - 2028.416054007841 Fraction[46] + Q0_var[46] = 0
 R464: Q0_var[46] + Lost_0_expr_46 = 1676.057662553893
 R465: Q0_var[46] + Q_hat_adjusted_46 = 2058.768596505982
 R466: Sold_0[46] - Q0_var[46] - Q1_var[46] + total_stock_second_period_46
   = 0
 R467: total_stock_second_period_46 + Lost_1_expr_46 = 382.7106559292545
 R468: Sold_1[46] + Left_1[46] - total_stock_second_period_46 = 0
 R469: - 700 Sold_0[46] - 700 Sold_1[46] + 700 Lost_0[46] + 700 Lost_1[46]
   + 300 Left_1[46] + profits_vars[46] = 0
 Fraction_lower_bound_47: Fraction[47] >= 0
 Fraction_upper_bound_47: Fraction[47] <= 1
 R472: 1.502350960657328 alphas[0] - 1.364582467079067 alphas[1]
   + 1.007000167908843 alphas[2] + 0.3985862233396426 alphas[3] - alphas[4]
   + f_var[47] = 0
 R473: - 2028.416054007841 Fraction[47] + Q0_var[47] = 0
 R474: Q0_var[47] + Lost_0_expr_47 = 902.1824135669464
 R475: Q0_var[47] + Q_hat_adjusted_47 = 1093.822610498864
 R476: Sold_0[47] - Q0_var[47] - Q1_var[47] + total_stock_second_period_47
   = 0
 R477: total_stock_second_period_47 + Lost_1_expr_47 = 191.6399181015776
 R478: Sold_1[47] + Left_1[47] - total_stock_second_period_47 = 0
 R479: - 700 Sold_0[47] - 700 Sold_1[47] + 700 Lost_0[47] + 700 Lost_1[47]
   + 300 Left_1[47] + profits_vars[47] = 0
 Fraction_lower_bound_48: Fraction[48] >= 0
 Fraction_upper_bound_48: Fraction[48] <= 1
 R482: 0.8173510907951662 alphas[0] - 0.526623881700362 alphas[1]
   + 1.146625172499899 alphas[2] - 0.5840656824573925 alphas[3] - alphas[4]
   + f_var[48] = 0
 R483: - 2028.416054007841 Fraction[48] + Q0_var[48] = 0
 R484: Q0_var[48] + Lost_0_expr_48 = 806.6974075638116
 R485: Q0_var[48] + Q_hat_adjusted_48 = 979.1215670775243
 R486: Sold_0[48] - Q0_var[48] - Q1_var[48] + total_stock_second_period_48
   = 0
 R487: total_stock_second_period_48 + Lost_1_expr_48 = 172.4238796732542
 R488: Sold_1[48] + Left_1[48] - total_stock_second_period_48 = 0
 R489: - 700 Sold_0[48] - 700 Sold_1[48] + 700 Lost_0[48] + 700 Lost_1[48]
   + 300 Left_1[48] + profits_vars[48] = 0
 Fraction_lower_bound_49: Fraction[49] >= 0
 Fraction_upper_bound_49: Fraction[49] <= 1
 R492: 0.6400155019933705 alphas[0] - 1.688611157892237 alphas[1]
   - 0.1002627207618077 alphas[2] - 1.970174987241885 alphas[3] - alphas[4]
   + f_var[49] = 0
 R493: - 2028.416054007841 Fraction[49] + Q0_var[49] = 0
 R494: Q0_var[49] + Lost_0_expr_49 = 1220.997436910324
 R495: Q0_var[49] + Q_hat_adjusted_49 = 1565.025855568255
 R496: Sold_0[49] - Q0_var[49] - Q1_var[49] + total_stock_second_period_49
   = 0
 R497: total_stock_second_period_49 + Lost_1_expr_49 = 344.0281403244772
 R498: Sold_1[49] + Left_1[49] - total_stock_second_period_49 = 0
 R499: - 700 Sold_0[49] - 700 Sold_1[49] + 700 Lost_0[49] + 700 Lost_1[49]
   + 300 Left_1[49] + profits_vars[49] = 0
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
 f_var[25] free
 f_var[26] free
 f_var[27] free
 f_var[28] free
 f_var[29] free
 f_var[30] free
 f_var[31] free
 f_var[32] free
 f_var[33] free
 f_var[34] free
 f_var[35] free
 f_var[36] free
 f_var[37] free
 f_var[38] free
 f_var[39] free
 f_var[40] free
 f_var[41] free
 f_var[42] free
 f_var[43] free
 f_var[44] free
 f_var[45] free
 f_var[46] free
 f_var[47] free
 f_var[48] free
 f_var[49] free
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
 Fraction[25] <= 1
 Fraction[26] <= 1
 Fraction[27] <= 1
 Fraction[28] <= 1
 Fraction[29] <= 1
 Fraction[30] <= 1
 Fraction[31] <= 1
 Fraction[32] <= 1
 Fraction[33] <= 1
 Fraction[34] <= 1
 Fraction[35] <= 1
 Fraction[36] <= 1
 Fraction[37] <= 1
 Fraction[38] <= 1
 Fraction[39] <= 1
 Fraction[40] <= 1
 Fraction[41] <= 1
 Fraction[42] <= 1
 Fraction[43] <= 1
 Fraction[44] <= 1
 Fraction[45] <= 1
 Fraction[46] <= 1
 Fraction[47] <= 1
 Fraction[48] <= 1
 Fraction[49] <= 1
 Q0_var[0] <= 2029.416054007841
 Q0_var[1] <= 2029.416054007841
 Q0_var[2] <= 2029.416054007841
 Q0_var[3] <= 2029.416054007841
 Q0_var[4] <= 2029.416054007841
 Q0_var[5] <= 2029.416054007841
 Q0_var[6] <= 2029.416054007841
 Q0_var[7] <= 2029.416054007841
 Q0_var[8] <= 2029.416054007841
 Q0_var[9] <= 2029.416054007841
 Q0_var[10] <= 2029.416054007841
 Q0_var[11] <= 2029.416054007841
 Q0_var[12] <= 2029.416054007841
 Q0_var[13] <= 2029.416054007841
 Q0_var[14] <= 2029.416054007841
 Q0_var[15] <= 2029.416054007841
 Q0_var[16] <= 2029.416054007841
 Q0_var[17] <= 2029.416054007841
 Q0_var[18] <= 2029.416054007841
 Q0_var[19] <= 2029.416054007841
 Q0_var[20] <= 2029.416054007841
 Q0_var[21] <= 2029.416054007841
 Q0_var[22] <= 2029.416054007841
 Q0_var[23] <= 2029.416054007841
 Q0_var[24] <= 2029.416054007841
 Q0_var[25] <= 2029.416054007841
 Q0_var[26] <= 2029.416054007841
 Q0_var[27] <= 2029.416054007841
 Q0_var[28] <= 2029.416054007841
 Q0_var[29] <= 2029.416054007841
 Q0_var[30] <= 2029.416054007841
 Q0_var[31] <= 2029.416054007841
 Q0_var[32] <= 2029.416054007841
 Q0_var[33] <= 2029.416054007841
 Q0_var[34] <= 2029.416054007841
 Q0_var[35] <= 2029.416054007841
 Q0_var[36] <= 2029.416054007841
 Q0_var[37] <= 2029.416054007841
 Q0_var[38] <= 2029.416054007841
 Q0_var[39] <= 2029.416054007841
 Q0_var[40] <= 2029.416054007841
 Q0_var[41] <= 2029.416054007841
 Q0_var[42] <= 2029.416054007841
 Q0_var[43] <= 2029.416054007841
 Q0_var[44] <= 2029.416054007841
 Q0_var[45] <= 2029.416054007841
 Q0_var[46] <= 2029.416054007841
 Q0_var[47] <= 2029.416054007841
 Q0_var[48] <= 2029.416054007841
 Q0_var[49] <= 2029.416054007841
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
 profits_vars[25] free
 profits_vars[26] free
 profits_vars[27] free
 profits_vars[28] free
 profits_vars[29] free
 profits_vars[30] free
 profits_vars[31] free
 profits_vars[32] free
 profits_vars[33] free
 profits_vars[34] free
 profits_vars[35] free
 profits_vars[36] free
 profits_vars[37] free
 profits_vars[38] free
 profits_vars[39] free
 profits_vars[40] free
 profits_vars[41] free
 profits_vars[42] free
 profits_vars[43] free
 profits_vars[44] free
 profits_vars[45] free
 profits_vars[46] free
 profits_vars[47] free
 profits_vars[48] free
 profits_vars[49] free
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
 Lost_0_expr_25 free
 Q_hat_adjusted_25 free
 Lost_1_expr_25 free
 Lost_0_expr_26 free
 Q_hat_adjusted_26 free
 Lost_1_expr_26 free
 Lost_0_expr_27 free
 Q_hat_adjusted_27 free
 Lost_1_expr_27 free
 Lost_0_expr_28 free
 Q_hat_adjusted_28 free
 Lost_1_expr_28 free
 Lost_0_expr_29 free
 Q_hat_adjusted_29 free
 Lost_1_expr_29 free
 Lost_0_expr_30 free
 Q_hat_adjusted_30 free
 Lost_1_expr_30 free
 Lost_0_expr_31 free
 Q_hat_adjusted_31 free
 Lost_1_expr_31 free
 Lost_0_expr_32 free
 Q_hat_adjusted_32 free
 Lost_1_expr_32 free
 Lost_0_expr_33 free
 Q_hat_adjusted_33 free
 Lost_1_expr_33 free
 Lost_0_expr_34 free
 Q_hat_adjusted_34 free
 Lost_1_expr_34 free
 Lost_0_expr_35 free
 Q_hat_adjusted_35 free
 Lost_1_expr_35 free
 Lost_0_expr_36 free
 Q_hat_adjusted_36 free
 Lost_1_expr_36 free
 Lost_0_expr_37 free
 Q_hat_adjusted_37 free
 Lost_1_expr_37 free
 Lost_0_expr_38 free
 Q_hat_adjusted_38 free
 Lost_1_expr_38 free
 Lost_0_expr_39 free
 Q_hat_adjusted_39 free
 Lost_1_expr_39 free
 Lost_0_expr_40 free
 Q_hat_adjusted_40 free
 Lost_1_expr_40 free
 Lost_0_expr_41 free
 Q_hat_adjusted_41 free
 Lost_1_expr_41 free
 Lost_0_expr_42 free
 Q_hat_adjusted_42 free
 Lost_1_expr_42 free
 Lost_0_expr_43 free
 Q_hat_adjusted_43 free
 Lost_1_expr_43 free
 Lost_0_expr_44 free
 Q_hat_adjusted_44 free
 Lost_1_expr_44 free
 Lost_0_expr_45 free
 Q_hat_adjusted_45 free
 Lost_1_expr_45 free
 Lost_0_expr_46 free
 Q_hat_adjusted_46 free
 Lost_1_expr_46 free
 Lost_0_expr_47 free
 Q_hat_adjusted_47 free
 Lost_1_expr_47 free
 Lost_0_expr_48 free
 Q_hat_adjusted_48 free
 Lost_1_expr_48 free
 Lost_0_expr_49 free
 Q_hat_adjusted_49 free
 Lost_1_expr_49 free
General Constraints
 GC0: Fraction[0] = LOGISTIC ( f_var[0] )
 min_sales_constr_0: Sold_0[0] = MIN ( Q0_var[0] , 985.5016692210908 )
 max_lost_constr_0: Lost_0[0] = MAX ( Lost_0_expr_0 , 0 )
 max_Q1_constr_0: Q1_var[0] = MAX ( Q_hat_adjusted_0 , 0 )
 min_sales2_constr_0: Sold_1[0] = MIN ( total_stock_second_period_0
   , 255.8979409970531 )
 max_lost2_constr_0: Lost_1[0] = MAX ( Lost_1_expr_0 , 0 )
 GC6: Fraction[1] = LOGISTIC ( f_var[1] )
 min_sales_constr_1: Sold_0[1] = MIN ( Q0_var[1] , 1570.720159318349 )
 max_lost_constr_1: Lost_0[1] = MAX ( Lost_0_expr_1 , 0 )
 max_Q1_constr_1: Q1_var[1] = MAX ( Q_hat_adjusted_1 , 0 )
 min_sales2_constr_1: Sold_1[1] = MIN ( total_stock_second_period_1
   , 448.1717444077298 )
 max_lost2_constr_1: Lost_1[1] = MAX ( Lost_1_expr_1 , 0 )
 GC12: Fraction[2] = LOGISTIC ( f_var[2] )
 min_sales_constr_2: Sold_0[2] = MIN ( Q0_var[2] , 1024.019920128306 )
 max_lost_constr_2: Lost_0[2] = MAX ( Lost_0_expr_2 , 0 )
 max_Q1_constr_2: Q1_var[2] = MAX ( Q_hat_adjusted_2 , 0 )
 min_sales2_constr_2: Sold_1[2] = MIN ( total_stock_second_period_2
   , 288.8759196462405 )
 max_lost2_constr_2: Lost_1[2] = MAX ( Lost_1_expr_2 , 0 )
 GC18: Fraction[3] = LOGISTIC ( f_var[3] )
 min_sales_constr_3: Sold_0[3] = MIN ( Q0_var[3] , 841.8614268697045 )
 max_lost_constr_3: Lost_0[3] = MAX ( Lost_0_expr_3 , 0 )
 max_Q1_constr_3: Q1_var[3] = MAX ( Q_hat_adjusted_3 , 0 )
 min_sales2_constr_3: Sold_1[3] = MIN ( total_stock_second_period_3
   , 189.8018387784145 )
 max_lost2_constr_3: Lost_1[3] = MAX ( Lost_1_expr_3 , 0 )
 GC24: Fraction[4] = LOGISTIC ( f_var[4] )
 min_sales_constr_4: Sold_0[4] = MIN ( Q0_var[4] , 944.6580682618797 )
 max_lost_constr_4: Lost_0[4] = MAX ( Lost_0_expr_4 , 0 )
 max_Q1_constr_4: Q1_var[4] = MAX ( Q_hat_adjusted_4 , 0 )
 min_sales2_constr_4: Sold_1[4] = MIN ( total_stock_second_period_4
   , 112.3393641955375 )
 max_lost2_constr_4: Lost_1[4] = MAX ( Lost_1_expr_4 , 0 )
 GC30: Fraction[5] = LOGISTIC ( f_var[5] )
 min_sales_constr_5: Sold_0[5] = MIN ( Q0_var[5] , 621.5387210066472 )
 max_lost_constr_5: Lost_0[5] = MAX ( Lost_0_expr_5 , 0 )
 max_Q1_constr_5: Q1_var[5] = MAX ( Q_hat_adjusted_5 , 0 )
 min_sales2_constr_5: Sold_1[5] = MIN ( total_stock_second_period_5
   , 133.6248157314834 )
 max_lost2_constr_5: Lost_1[5] = MAX ( Lost_1_expr_5 , 0 )
 GC36: Fraction[6] = LOGISTIC ( f_var[6] )
 min_sales_constr_6: Sold_0[6] = MIN ( Q0_var[6] , 1847.22968764704 )
 max_lost_constr_6: Lost_0[6] = MAX ( Lost_0_expr_6 , 0 )
 max_Q1_constr_6: Q1_var[6] = MAX ( Q_hat_adjusted_6 , 0 )
 min_sales2_constr_6: Sold_1[6] = MIN ( total_stock_second_period_6
   , 439.6963867492836 )
 max_lost2_constr_6: Lost_1[6] = MAX ( Lost_1_expr_6 , 0 )
 GC42: Fraction[7] = LOGISTIC ( f_var[7] )
 min_sales_constr_7: Sold_0[7] = MIN ( Q0_var[7] , 1042.381783858741 )
 max_lost_constr_7: Lost_0[7] = MAX ( Lost_0_expr_7 , 0 )
 max_Q1_constr_7: Q1_var[7] = MAX ( Q_hat_adjusted_7 , 0 )
 min_sales2_constr_7: Sold_1[7] = MIN ( total_stock_second_period_7
   , 326.8484222780227 )
 max_lost2_constr_7: Lost_1[7] = MAX ( Lost_1_expr_7 , 0 )
 GC48: Fraction[8] = LOGISTIC ( f_var[8] )
 min_sales_constr_8: Sold_0[8] = MIN ( Q0_var[8] , 1257.903530849573 )
 max_lost_constr_8: Lost_0[8] = MAX ( Lost_0_expr_8 , 0 )
 max_Q1_constr_8: Q1_var[8] = MAX ( Q_hat_adjusted_8 , 0 )
 min_sales2_constr_8: Sold_1[8] = MIN ( total_stock_second_period_8
   , 368.2896055543611 )
 max_lost2_constr_8: Lost_1[8] = MAX ( Lost_1_expr_8 , 0 )
 GC54: Fraction[9] = LOGISTIC ( f_var[9] )
 min_sales_constr_9: Sold_0[9] = MIN ( Q0_var[9] , 2114.35714324075 )
 max_lost_constr_9: Lost_0[9] = MAX ( Lost_0_expr_9 , 0 )
 max_Q1_constr_9: Q1_var[9] = MAX ( Q_hat_adjusted_9 , 0 )
 min_sales2_constr_9: Sold_1[9] = MIN ( total_stock_second_period_9
   , 548.3356465201711 )
 max_lost2_constr_9: Lost_1[9] = MAX ( Lost_1_expr_9 , 0 )
 GC60: Fraction[10] = LOGISTIC ( f_var[10] )
 min_sales_constr_10: Sold_0[10] = MIN ( Q0_var[10] , 2776.400662008934 )
 max_lost_constr_10: Lost_0[10] = MAX ( Lost_0_expr_10 , 0 )
 max_Q1_constr_10: Q1_var[10] = MAX ( Q_hat_adjusted_10 , 0 )
 min_sales2_constr_10: Sold_1[10] = MIN ( total_stock_second_period_10
   , 595.1694736983521 )
 max_lost2_constr_10: Lost_1[10] = MAX ( Lost_1_expr_10 , 0 )
 GC66: Fraction[11] = LOGISTIC ( f_var[11] )
 min_sales_constr_11: Sold_0[11] = MIN ( Q0_var[11] , 1085.780691702743 )
 max_lost_constr_11: Lost_0[11] = MAX ( Lost_0_expr_11 , 0 )
 max_Q1_constr_11: Q1_var[11] = MAX ( Q_hat_adjusted_11 , 0 )
 min_sales2_constr_11: Sold_1[11] = MIN ( total_stock_second_period_11
   , 208.4484921991337 )
 max_lost2_constr_11: Lost_1[11] = MAX ( Lost_1_expr_11 , 0 )
 GC72: Fraction[12] = LOGISTIC ( f_var[12] )
 min_sales_constr_12: Sold_0[12] = MIN ( Q0_var[12] , 1803.127894967017 )
 max_lost_constr_12: Lost_0[12] = MAX ( Lost_0_expr_12 , 0 )
 max_Q1_constr_12: Q1_var[12] = MAX ( Q_hat_adjusted_12 , 0 )
 min_sales2_constr_12: Sold_1[12] = MIN ( total_stock_second_period_12
   , 431.5391015500504 )
 max_lost2_constr_12: Lost_1[12] = MAX ( Lost_1_expr_12 , 0 )
 GC78: Fraction[13] = LOGISTIC ( f_var[13] )
 min_sales_constr_13: Sold_0[13] = MIN ( Q0_var[13] , 1026.645354231385 )
 max_lost_constr_13: Lost_0[13] = MAX ( Lost_0_expr_13 , 0 )
 max_Q1_constr_13: Q1_var[13] = MAX ( Q_hat_adjusted_13 , 0 )
 min_sales2_constr_13: Sold_1[13] = MIN ( total_stock_second_period_13
   , 231.6611883685753 )
 max_lost2_constr_13: Lost_1[13] = MAX ( Lost_1_expr_13 , 0 )
 GC84: Fraction[14] = LOGISTIC ( f_var[14] )
 min_sales_constr_14: Sold_0[14] = MIN ( Q0_var[14] , 487.4053591546971 )
 max_lost_constr_14: Lost_0[14] = MAX ( Lost_0_expr_14 , 0 )
 max_Q1_constr_14: Q1_var[14] = MAX ( Q_hat_adjusted_14 , 0 )
 min_sales2_constr_14: Sold_1[14] = MIN ( total_stock_second_period_14
   , 110.32549913407 )
 max_lost2_constr_14: Lost_1[14] = MAX ( Lost_1_expr_14 , 0 )
 GC90: Fraction[15] = LOGISTIC ( f_var[15] )
 min_sales_constr_15: Sold_0[15] = MIN ( Q0_var[15] , 1929.829545184692 )
 max_lost_constr_15: Lost_0[15] = MAX ( Lost_0_expr_15 , 0 )
 max_Q1_constr_15: Q1_var[15] = MAX ( Q_hat_adjusted_15 , 0 )
 min_sales2_constr_15: Sold_1[15] = MIN ( total_stock_second_period_15
   , 479.1893269180707 )
 max_lost2_constr_15: Lost_1[15] = MAX ( Lost_1_expr_15 , 0 )
 GC96: Fraction[16] = LOGISTIC ( f_var[16] )
 min_sales_constr_16: Sold_0[16] = MIN ( Q0_var[16] , 877.6680076190777 )
 max_lost_constr_16: Lost_0[16] = MAX ( Lost_0_expr_16 , 0 )
 max_Q1_constr_16: Q1_var[16] = MAX ( Q_hat_adjusted_16 , 0 )
 min_sales2_constr_16: Sold_1[16] = MIN ( total_stock_second_period_16
   , 260.0085754112481 )
 max_lost2_constr_16: Lost_1[16] = MAX ( Lost_1_expr_16 , 0 )
 GC102: Fraction[17] = LOGISTIC ( f_var[17] )
 min_sales_constr_17: Sold_0[17] = MIN ( Q0_var[17] , 1000.389772420226 )
 max_lost_constr_17: Lost_0[17] = MAX ( Lost_0_expr_17 , 0 )
 max_Q1_constr_17: Q1_var[17] = MAX ( Q_hat_adjusted_17 , 0 )
 min_sales2_constr_17: Sold_1[17] = MIN ( total_stock_second_period_17
   , 291.731947085949 )
 max_lost2_constr_17: Lost_1[17] = MAX ( Lost_1_expr_17 , 0 )
 GC108: Fraction[18] = LOGISTIC ( f_var[18] )
 min_sales_constr_18: Sold_0[18] = MIN ( Q0_var[18] , 1663.667313796551 )
 max_lost_constr_18: Lost_0[18] = MAX ( Lost_0_expr_18 , 0 )
 max_Q1_constr_18: Q1_var[18] = MAX ( Q_hat_adjusted_18 , 0 )
 min_sales2_constr_18: Sold_1[18] = MIN ( total_stock_second_period_18
   , 394.1585567768029 )
 max_lost2_constr_18: Lost_1[18] = MAX ( Lost_1_expr_18 , 0 )
 GC114: Fraction[19] = LOGISTIC ( f_var[19] )
 min_sales_constr_19: Sold_0[19] = MIN ( Q0_var[19] , 1980.356078446152 )
 max_lost_constr_19: Lost_0[19] = MAX ( Lost_0_expr_19 , 0 )
 max_Q1_constr_19: Q1_var[19] = MAX ( Q_hat_adjusted_19 , 0 )
 min_sales2_constr_19: Sold_1[19] = MIN ( total_stock_second_period_19
   , 515.5242276180952 )
 max_lost2_constr_19: Lost_1[19] = MAX ( Lost_1_expr_19 , 0 )
 GC120: Fraction[20] = LOGISTIC ( f_var[20] )
 min_sales_constr_20: Sold_0[20] = MIN ( Q0_var[20] , 1475.395085876261 )
 max_lost_constr_20: Lost_0[20] = MAX ( Lost_0_expr_20 , 0 )
 max_Q1_constr_20: Q1_var[20] = MAX ( Q_hat_adjusted_20 , 0 )
 min_sales2_constr_20: Sold_1[20] = MIN ( total_stock_second_period_20
   , 414.4909217955744 )
 max_lost2_constr_20: Lost_1[20] = MAX ( Lost_1_expr_20 , 0 )
 GC126: Fraction[21] = LOGISTIC ( f_var[21] )
 min_sales_constr_21: Sold_0[21] = MIN ( Q0_var[21] , 1997.067163955949 )
 max_lost_constr_21: Lost_0[21] = MAX ( Lost_0_expr_21 , 0 )
 max_Q1_constr_21: Q1_var[21] = MAX ( Q_hat_adjusted_21 , 0 )
 min_sales2_constr_21: Sold_1[21] = MIN ( total_stock_second_period_21
   , 536.3253277236977 )
 max_lost2_constr_21: Lost_1[21] = MAX ( Lost_1_expr_21 , 0 )
 GC132: Fraction[22] = LOGISTIC ( f_var[22] )
 min_sales_constr_22: Sold_0[22] = MIN ( Q0_var[22] , 1091.077127926938 )
 max_lost_constr_22: Lost_0[22] = MAX ( Lost_0_expr_22 , 0 )
 max_Q1_constr_22: Q1_var[22] = MAX ( Q_hat_adjusted_22 , 0 )
 min_sales2_constr_22: Sold_1[22] = MIN ( total_stock_second_period_22
   , 236.7710174077933 )
 max_lost2_constr_22: Lost_1[22] = MAX ( Lost_1_expr_22 , 0 )
 GC138: Fraction[23] = LOGISTIC ( f_var[23] )
 min_sales_constr_23: Sold_0[23] = MIN ( Q0_var[23] , 2145.597831988234 )
 max_lost_constr_23: Lost_0[23] = MAX ( Lost_0_expr_23 , 0 )
 max_Q1_constr_23: Q1_var[23] = MAX ( Q_hat_adjusted_23 , 0 )
 min_sales2_constr_23: Sold_1[23] = MIN ( total_stock_second_period_23
   , 499.0234169562818 )
 max_lost2_constr_23: Lost_1[23] = MAX ( Lost_1_expr_23 , 0 )
 GC144: Fraction[24] = LOGISTIC ( f_var[24] )
 min_sales_constr_24: Sold_0[24] = MIN ( Q0_var[24] , 803.8390560387079 )
 max_lost_constr_24: Lost_0[24] = MAX ( Lost_0_expr_24 , 0 )
 max_Q1_constr_24: Q1_var[24] = MAX ( Q_hat_adjusted_24 , 0 )
 min_sales2_constr_24: Sold_1[24] = MIN ( total_stock_second_period_24
   , 192.8179721529758 )
 max_lost2_constr_24: Lost_1[24] = MAX ( Lost_1_expr_24 , 0 )
 GC150: Fraction[25] = LOGISTIC ( f_var[25] )
 min_sales_constr_25: Sold_0[25] = MIN ( Q0_var[25] , 2289.88115278488 )
 max_lost_constr_25: Lost_0[25] = MAX ( Lost_0_expr_25 , 0 )
 max_Q1_constr_25: Q1_var[25] = MAX ( Q_hat_adjusted_25 , 0 )
 min_sales2_constr_25: Sold_1[25] = MIN ( total_stock_second_period_25
   , 576.3958284859723 )
 max_lost2_constr_25: Lost_1[25] = MAX ( Lost_1_expr_25 , 0 )
 GC156: Fraction[26] = LOGISTIC ( f_var[26] )
 min_sales_constr_26: Sold_0[26] = MIN ( Q0_var[26] , 1936.480774365079 )
 max_lost_constr_26: Lost_0[26] = MAX ( Lost_0_expr_26 , 0 )
 max_Q1_constr_26: Q1_var[26] = MAX ( Q_hat_adjusted_26 , 0 )
 min_sales2_constr_26: Sold_1[26] = MIN ( total_stock_second_period_26
   , 443.7441381939076 )
 max_lost2_constr_26: Lost_1[26] = MAX ( Lost_1_expr_26 , 0 )
 GC162: Fraction[27] = LOGISTIC ( f_var[27] )
 min_sales_constr_27: Sold_0[27] = MIN ( Q0_var[27] , 888.6520263497487 )
 max_lost_constr_27: Lost_0[27] = MAX ( Lost_0_expr_27 , 0 )
 max_Q1_constr_27: Q1_var[27] = MAX ( Q_hat_adjusted_27 , 0 )
 min_sales2_constr_27: Sold_1[27] = MIN ( total_stock_second_period_27
   , 207.7538385567793 )
 max_lost2_constr_27: Lost_1[27] = MAX ( Lost_1_expr_27 , 0 )
 GC168: Fraction[28] = LOGISTIC ( f_var[28] )
 min_sales_constr_28: Sold_0[28] = MIN ( Q0_var[28] , 2420.221455590129 )
 max_lost_constr_28: Lost_0[28] = MAX ( Lost_0_expr_28 , 0 )
 max_Q1_constr_28: Q1_var[28] = MAX ( Q_hat_adjusted_28 , 0 )
 min_sales2_constr_28: Sold_1[28] = MIN ( total_stock_second_period_28
   , 573.6852952444622 )
 max_lost2_constr_28: Lost_1[28] = MAX ( Lost_1_expr_28 , 0 )
 GC174: Fraction[29] = LOGISTIC ( f_var[29] )
 min_sales_constr_29: Sold_0[29] = MIN ( Q0_var[29] , 1415.696311696586 )
 max_lost_constr_29: Lost_0[29] = MAX ( Lost_0_expr_29 , 0 )
 max_Q1_constr_29: Q1_var[29] = MAX ( Q_hat_adjusted_29 , 0 )
 min_sales2_constr_29: Sold_1[29] = MIN ( total_stock_second_period_29
   , 465.4279033850789 )
 max_lost2_constr_29: Lost_1[29] = MAX ( Lost_1_expr_29 , 0 )
 GC180: Fraction[30] = LOGISTIC ( f_var[30] )
 min_sales_constr_30: Sold_0[30] = MIN ( Q0_var[30] , 869.7617671527879 )
 max_lost_constr_30: Lost_0[30] = MAX ( Lost_0_expr_30 , 0 )
 max_Q1_constr_30: Q1_var[30] = MAX ( Q_hat_adjusted_30 , 0 )
 min_sales2_constr_30: Sold_1[30] = MIN ( total_stock_second_period_30
   , 226.9708212975124 )
 max_lost2_constr_30: Lost_1[30] = MAX ( Lost_1_expr_30 , 0 )
 GC186: Fraction[31] = LOGISTIC ( f_var[31] )
 min_sales_constr_31: Sold_0[31] = MIN ( Q0_var[31] , 953.6834708503509 )
 max_lost_constr_31: Lost_0[31] = MAX ( Lost_0_expr_31 , 0 )
 max_Q1_constr_31: Q1_var[31] = MAX ( Q_hat_adjusted_31 , 0 )
 min_sales2_constr_31: Sold_1[31] = MIN ( total_stock_second_period_31
   , 206.6559886837406 )
 max_lost2_constr_31: Lost_1[31] = MAX ( Lost_1_expr_31 , 0 )
 GC192: Fraction[32] = LOGISTIC ( f_var[32] )
 min_sales_constr_32: Sold_0[32] = MIN ( Q0_var[32] , 1410.101183009175 )
 max_lost_constr_32: Lost_0[32] = MAX ( Lost_0_expr_32 , 0 )
 max_Q1_constr_32: Q1_var[32] = MAX ( Q_hat_adjusted_32 , 0 )
 min_sales2_constr_32: Sold_1[32] = MIN ( total_stock_second_period_32
   , 359.1003569653316 )
 max_lost2_constr_32: Lost_1[32] = MAX ( Lost_1_expr_32 , 0 )
 GC198: Fraction[33] = LOGISTIC ( f_var[33] )
 min_sales_constr_33: Sold_0[33] = MIN ( Q0_var[33] , 553.121104136178 )
 max_lost_constr_33: Lost_0[33] = MAX ( Lost_0_expr_33 , 0 )
 max_Q1_constr_33: Q1_var[33] = MAX ( Q_hat_adjusted_33 , 0 )
 min_sales2_constr_33: Sold_1[33] = MIN ( total_stock_second_period_33
   , 112.8313590284054 )
 max_lost2_constr_33: Lost_1[33] = MAX ( Lost_1_expr_33 , 0 )
 GC204: Fraction[34] = LOGISTIC ( f_var[34] )
 min_sales_constr_34: Sold_0[34] = MIN ( Q0_var[34] , 941.8925750063599 )
 max_lost_constr_34: Lost_0[34] = MAX ( Lost_0_expr_34 , 0 )
 max_Q1_constr_34: Q1_var[34] = MAX ( Q_hat_adjusted_34 , 0 )
 min_sales2_constr_34: Sold_1[34] = MIN ( total_stock_second_period_34
   , 203.7350377205705 )
 max_lost2_constr_34: Lost_1[34] = MAX ( Lost_1_expr_34 , 0 )
 GC210: Fraction[35] = LOGISTIC ( f_var[35] )
 min_sales_constr_35: Sold_0[35] = MIN ( Q0_var[35] , 1059.166411412334 )
 max_lost_constr_35: Lost_0[35] = MAX ( Lost_0_expr_35 , 0 )
 max_Q1_constr_35: Q1_var[35] = MAX ( Q_hat_adjusted_35 , 0 )
 min_sales2_constr_35: Sold_1[35] = MIN ( total_stock_second_period_35
   , 312.3427343757531 )
 max_lost2_constr_35: Lost_1[35] = MAX ( Lost_1_expr_35 , 0 )
 GC216: Fraction[36] = LOGISTIC ( f_var[36] )
 min_sales_constr_36: Sold_0[36] = MIN ( Q0_var[36] , 999.2920081744734 )
 max_lost_constr_36: Lost_0[36] = MAX ( Lost_0_expr_36 , 0 )
 max_Q1_constr_36: Q1_var[36] = MAX ( Q_hat_adjusted_36 , 0 )
 min_sales2_constr_36: Sold_1[36] = MIN ( total_stock_second_period_36
   , 287.0849901671128 )
 max_lost2_constr_36: Lost_1[36] = MAX ( Lost_1_expr_36 , 0 )
 GC222: Fraction[37] = LOGISTIC ( f_var[37] )
 min_sales_constr_37: Sold_0[37] = MIN ( Q0_var[37] , 1218.506327701144 )
 max_lost_constr_37: Lost_0[37] = MAX ( Lost_0_expr_37 , 0 )
 max_Q1_constr_37: Q1_var[37] = MAX ( Q_hat_adjusted_37 , 0 )
 min_sales2_constr_37: Sold_1[37] = MIN ( total_stock_second_period_37
   , 331.7877121824054 )
 max_lost2_constr_37: Lost_1[37] = MAX ( Lost_1_expr_37 , 0 )
 GC228: Fraction[38] = LOGISTIC ( f_var[38] )
 min_sales_constr_38: Sold_0[38] = MIN ( Q0_var[38] , 1324.417089681451 )
 max_lost_constr_38: Lost_0[38] = MAX ( Lost_0_expr_38 , 0 )
 max_Q1_constr_38: Q1_var[38] = MAX ( Q_hat_adjusted_38 , 0 )
 min_sales2_constr_38: Sold_1[38] = MIN ( total_stock_second_period_38
   , 238.8143531473638 )
 max_lost2_constr_38: Lost_1[38] = MAX ( Lost_1_expr_38 , 0 )
 GC234: Fraction[39] = LOGISTIC ( f_var[39] )
 min_sales_constr_39: Sold_0[39] = MIN ( Q0_var[39] , 1600.632086988877 )
 max_lost_constr_39: Lost_0[39] = MAX ( Lost_0_expr_39 , 0 )
 max_Q1_constr_39: Q1_var[39] = MAX ( Q_hat_adjusted_39 , 0 )
 min_sales2_constr_39: Sold_1[39] = MIN ( total_stock_second_period_39
   , 393.3921732290844 )
 max_lost2_constr_39: Lost_1[39] = MAX ( Lost_1_expr_39 , 0 )
 GC240: Fraction[40] = LOGISTIC ( f_var[40] )
 min_sales_constr_40: Sold_0[40] = MIN ( Q0_var[40] , 2091.203152144836 )
 max_lost_constr_40: Lost_0[40] = MAX ( Lost_0_expr_40 , 0 )
 max_Q1_constr_40: Q1_var[40] = MAX ( Q_hat_adjusted_40 , 0 )
 min_sales2_constr_40: Sold_1[40] = MIN ( total_stock_second_period_40
   , 531.9278029616157 )
 max_lost2_constr_40: Lost_1[40] = MAX ( Lost_1_expr_40 , 0 )
 GC246: Fraction[41] = LOGISTIC ( f_var[41] )
 min_sales_constr_41: Sold_0[41] = MIN ( Q0_var[41] , 640.3005463220409 )
 max_lost_constr_41: Lost_0[41] = MAX ( Lost_0_expr_41 , 0 )
 max_Q1_constr_41: Q1_var[41] = MAX ( Q_hat_adjusted_41 , 0 )
 min_sales2_constr_41: Sold_1[41] = MIN ( total_stock_second_period_41
   , 158.7659279810111 )
 max_lost2_constr_41: Lost_1[41] = MAX ( Lost_1_expr_41 , 0 )
 GC252: Fraction[42] = LOGISTIC ( f_var[42] )
 min_sales_constr_42: Sold_0[42] = MIN ( Q0_var[42] , 1809.380251590136 )
 max_lost_constr_42: Lost_0[42] = MAX ( Lost_0_expr_42 , 0 )
 max_Q1_constr_42: Q1_var[42] = MAX ( Q_hat_adjusted_42 , 0 )
 min_sales2_constr_42: Sold_1[42] = MIN ( total_stock_second_period_42
   , 358.6895535770571 )
 max_lost2_constr_42: Lost_1[42] = MAX ( Lost_1_expr_42 , 0 )
 GC258: Fraction[43] = LOGISTIC ( f_var[43] )
 min_sales_constr_43: Sold_0[43] = MIN ( Q0_var[43] , 657.0729437534862 )
 max_lost_constr_43: Lost_0[43] = MAX ( Lost_0_expr_43 , 0 )
 max_Q1_constr_43: Q1_var[43] = MAX ( Q_hat_adjusted_43 , 0 )
 min_sales2_constr_43: Sold_1[43] = MIN ( total_stock_second_period_43
   , 166.034053171934 )
 max_lost2_constr_43: Lost_1[43] = MAX ( Lost_1_expr_43 , 0 )
 GC264: Fraction[44] = LOGISTIC ( f_var[44] )
 min_sales_constr_44: Sold_0[44] = MIN ( Q0_var[44] , 1895.918504047045 )
 max_lost_constr_44: Lost_0[44] = MAX ( Lost_0_expr_44 , 0 )
 max_Q1_constr_44: Q1_var[44] = MAX ( Q_hat_adjusted_44 , 0 )
 min_sales2_constr_44: Sold_1[44] = MIN ( total_stock_second_period_44
   , 458.4298405962969 )
 max_lost2_constr_44: Lost_1[44] = MAX ( Lost_1_expr_44 , 0 )
 GC270: Fraction[45] = LOGISTIC ( f_var[45] )
 min_sales_constr_45: Sold_0[45] = MIN ( Q0_var[45] , 1052.136144779188 )
 max_lost_constr_45: Lost_0[45] = MAX ( Lost_0_expr_45 , 0 )
 max_Q1_constr_45: Q1_var[45] = MAX ( Q_hat_adjusted_45 , 0 )
 min_sales2_constr_45: Sold_1[45] = MIN ( total_stock_second_period_45
   , 298.0298514036469 )
 max_lost2_constr_45: Lost_1[45] = MAX ( Lost_1_expr_45 , 0 )
 GC276: Fraction[46] = LOGISTIC ( f_var[46] )
 min_sales_constr_46: Sold_0[46] = MIN ( Q0_var[46] , 1676.057662553893 )
 max_lost_constr_46: Lost_0[46] = MAX ( Lost_0_expr_46 , 0 )
 max_Q1_constr_46: Q1_var[46] = MAX ( Q_hat_adjusted_46 , 0 )
 min_sales2_constr_46: Sold_1[46] = MIN ( total_stock_second_period_46
   , 382.7106559292545 )
 max_lost2_constr_46: Lost_1[46] = MAX ( Lost_1_expr_46 , 0 )
 GC282: Fraction[47] = LOGISTIC ( f_var[47] )
 min_sales_constr_47: Sold_0[47] = MIN ( Q0_var[47] , 902.1824135669464 )
 max_lost_constr_47: Lost_0[47] = MAX ( Lost_0_expr_47 , 0 )
 max_Q1_constr_47: Q1_var[47] = MAX ( Q_hat_adjusted_47 , 0 )
 min_sales2_constr_47: Sold_1[47] = MIN ( total_stock_second_period_47
   , 191.6399181015776 )
 max_lost2_constr_47: Lost_1[47] = MAX ( Lost_1_expr_47 , 0 )
 GC288: Fraction[48] = LOGISTIC ( f_var[48] )
 min_sales_constr_48: Sold_0[48] = MIN ( Q0_var[48] , 806.6974075638116 )
 max_lost_constr_48: Lost_0[48] = MAX ( Lost_0_expr_48 , 0 )
 max_Q1_constr_48: Q1_var[48] = MAX ( Q_hat_adjusted_48 , 0 )
 min_sales2_constr_48: Sold_1[48] = MIN ( total_stock_second_period_48
   , 172.4238796732542 )
 max_lost2_constr_48: Lost_1[48] = MAX ( Lost_1_expr_48 , 0 )
 GC294: Fraction[49] = LOGISTIC ( f_var[49] )
 min_sales_constr_49: Sold_0[49] = MIN ( Q0_var[49] , 1220.997436910324 )
 max_lost_constr_49: Lost_0[49] = MAX ( Lost_0_expr_49 , 0 )
 max_Q1_constr_49: Q1_var[49] = MAX ( Q_hat_adjusted_49 , 0 )
 min_sales2_constr_49: Sold_1[49] = MIN ( total_stock_second_period_49
   , 344.0281403244772 )
 max_lost2_constr_49: Lost_1[49] = MAX ( Lost_1_expr_49 , 0 )
End
