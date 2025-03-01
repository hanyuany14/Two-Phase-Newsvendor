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
   + profits_vars[29]
Subject To
 Fraction_lower_bound_0: Fraction[0] >= 0
 Fraction_upper_bound_0: Fraction[0] <= 1
 R2: - 2385.699259840349 Fraction[0] + Q0_var[0] = 0
 R3: Q0_var[0] + Lost_0_expr_0 = 1237.106505172745
 R4: Q0_var[0] + Q_hat_adjusted_0 = 1631.055584105667
 R5: Sold_0[0] - Q0_var[0] - Q1_var[0] + total_stock_second_period_0 = 0
 R6: total_stock_second_period_0 + Lost_1_expr_0 = 307.0146413669897
 R7: Sold_1[0] + Left_1[0] - total_stock_second_period_0 = 0
 R8: - 700 Sold_0[0] - 700 Sold_1[0] + 700 Lost_0[0] + 700 Lost_1[0]
   + 300 Left_1[0] + profits_vars[0] = 0
 Fraction_lower_bound_1: Fraction[1] >= 0
 Fraction_upper_bound_1: Fraction[1] <= 1
 R11: - 2385.699259840349 Fraction[1] + Q0_var[1] = 0
 R12: Q0_var[1] + Lost_0_expr_1 = 900.4115652482144
 R13: Q0_var[1] + Q_hat_adjusted_1 = 862.6318107807184
 R14: Sold_0[1] - Q0_var[1] - Q1_var[1] + total_stock_second_period_1 = 0
 R15: total_stock_second_period_1 + Lost_1_expr_1 = 19.24180288226607
 R16: Sold_1[1] + Left_1[1] - total_stock_second_period_1 = 0
 R17: - 700 Sold_0[1] - 700 Sold_1[1] + 700 Lost_0[1] + 700 Lost_1[1]
   + 300 Left_1[1] + profits_vars[1] = 0
 Fraction_lower_bound_2: Fraction[2] >= 0
 Fraction_upper_bound_2: Fraction[2] <= 1
 R20: - 2385.699259840349 Fraction[2] + Q0_var[2] = 0
 R21: Q0_var[2] + Lost_0_expr_2 = 348.3813833935464
 R22: Q0_var[2] + Q_hat_adjusted_2 = 522.3031349709138
 R23: Sold_0[2] - Q0_var[2] - Q1_var[2] + total_stock_second_period_2 = 0
 R24: total_stock_second_period_2 + Lost_1_expr_2 = 169.8599237814332
 R25: Sold_1[2] + Left_1[2] - total_stock_second_period_2 = 0
 R26: - 700 Sold_0[2] - 700 Sold_1[2] + 700 Lost_0[2] + 700 Lost_1[2]
   + 300 Left_1[2] + profits_vars[2] = 0
 Fraction_lower_bound_3: Fraction[3] >= 0
 Fraction_upper_bound_3: Fraction[3] <= 1
 R29: - 2385.699259840349 Fraction[3] + Q0_var[3] = 0
 R30: Q0_var[3] + Lost_0_expr_3 = 1576.839725251233
 R31: Q0_var[3] + Q_hat_adjusted_3 = 2315.096200269158
 R32: Sold_0[3] - Q0_var[3] - Q1_var[3] + total_stock_second_period_3 = 0
 R33: total_stock_second_period_3 + Lost_1_expr_3 = 765.0737567426161
 R34: Sold_1[3] + Left_1[3] - total_stock_second_period_3 = 0
 R35: - 700 Sold_0[3] - 700 Sold_1[3] + 700 Lost_0[3] + 700 Lost_1[3]
   + 300 Left_1[3] + profits_vars[3] = 0
 Fraction_lower_bound_4: Fraction[4] >= 0
 Fraction_upper_bound_4: Fraction[4] <= 1
 R38: - 2385.699259840349 Fraction[4] + Q0_var[4] = 0
 R39: Q0_var[4] + Lost_0_expr_4 = 1687.746268160732
 R40: Q0_var[4] + Q_hat_adjusted_4 = 2071.686250124366
 R41: Sold_0[4] - Q0_var[4] - Q1_var[4] + total_stock_second_period_4 = 0
 R42: total_stock_second_period_4 + Lost_1_expr_4 = 410.9698184910955
 R43: Sold_1[4] + Left_1[4] - total_stock_second_period_4 = 0
 R44: - 700 Sold_0[4] - 700 Sold_1[4] + 700 Lost_0[4] + 700 Lost_1[4]
   + 300 Left_1[4] + profits_vars[4] = 0
 Fraction_lower_bound_5: Fraction[5] >= 0
 Fraction_upper_bound_5: Fraction[5] <= 1
 R47: - 2385.699259840349 Fraction[5] + Q0_var[5] = 0
 R48: Q0_var[5] + Lost_0_expr_5 = 2443.617560988295
 R49: Q0_var[5] + Q_hat_adjusted_5 = 3071.137774579419
 R50: Sold_0[5] - Q0_var[5] - Q1_var[5] + total_stock_second_period_5 = 0
 R51: total_stock_second_period_5 + Lost_1_expr_5 = 622.232022253729
 R52: Sold_1[5] + Left_1[5] - total_stock_second_period_5 = 0
 R53: - 700 Sold_0[5] - 700 Sold_1[5] + 700 Lost_0[5] + 700 Lost_1[5]
   + 300 Left_1[5] + profits_vars[5] = 0
 Fraction_lower_bound_6: Fraction[6] >= 0
 Fraction_upper_bound_6: Fraction[6] <= 1
 R56: - 2385.699259840349 Fraction[6] + Q0_var[6] = 0
 R57: Q0_var[6] + Lost_0_expr_6 = 2004.656042059087
 R58: Q0_var[6] + Q_hat_adjusted_6 = 2432.380118898652
 R59: Sold_0[6] - Q0_var[6] - Q1_var[6] + total_stock_second_period_6 = 0
 R60: total_stock_second_period_6 + Lost_1_expr_6 = 427.178256319511
 R61: Sold_1[6] + Left_1[6] - total_stock_second_period_6 = 0
 R62: - 700 Sold_0[6] - 700 Sold_1[6] + 700 Lost_0[6] + 700 Lost_1[6]
   + 300 Left_1[6] + profits_vars[6] = 0
 Fraction_lower_bound_7: Fraction[7] >= 0
 Fraction_upper_bound_7: Fraction[7] <= 1
 R65: - 2385.699259840349 Fraction[7] + Q0_var[7] = 0
 R66: Q0_var[7] + Lost_0_expr_7 = 984.3984055832955
 R67: Q0_var[7] + Q_hat_adjusted_7 = 1510.804362399492
 R68: Sold_0[7] - Q0_var[7] - Q1_var[7] + total_stock_second_period_7 = 0
 R69: total_stock_second_period_7 + Lost_1_expr_7 = 349.8420422049586
 R70: Sold_1[7] + Left_1[7] - total_stock_second_period_7 = 0
 R71: - 700 Sold_0[7] - 700 Sold_1[7] + 700 Lost_0[7] + 700 Lost_1[7]
   + 300 Left_1[7] + profits_vars[7] = 0
 Fraction_lower_bound_8: Fraction[8] >= 0
 Fraction_upper_bound_8: Fraction[8] <= 1
 R74: - 2385.699259840349 Fraction[8] + Q0_var[8] = 0
 R75: Q0_var[8] + Lost_0_expr_8 = 1302.571392831954
 R76: Q0_var[8] + Q_hat_adjusted_8 = 1847.122924383442
 R77: Sold_0[8] - Q0_var[8] - Q1_var[8] + total_stock_second_period_8 = 0
 R78: total_stock_second_period_8 + Lost_1_expr_8 = 443.5751957473497
 R79: Sold_1[8] + Left_1[8] - total_stock_second_period_8 = 0
 R80: - 700 Sold_0[8] - 700 Sold_1[8] + 700 Lost_0[8] + 700 Lost_1[8]
   + 300 Left_1[8] + profits_vars[8] = 0
 Fraction_lower_bound_9: Fraction[9] >= 0
 Fraction_upper_bound_9: Fraction[9] <= 1
 R83: - 2385.699259840349 Fraction[9] + Q0_var[9] = 0
 R84: Q0_var[9] + Lost_0_expr_9 = 2170.16072797647
 R85: Q0_var[9] + Q_hat_adjusted_9 = 2682.460652802506
 R86: Sold_0[9] - Q0_var[9] - Q1_var[9] + total_stock_second_period_9 = 0
 R87: total_stock_second_period_9 + Lost_1_expr_9 = 387.8098237156327
 R88: Sold_1[9] + Left_1[9] - total_stock_second_period_9 = 0
 R89: - 700 Sold_0[9] - 700 Sold_1[9] + 700 Lost_0[9] + 700 Lost_1[9]
   + 300 Left_1[9] + profits_vars[9] = 0
 Fraction_lower_bound_10: Fraction[10] >= 0
 Fraction_upper_bound_10: Fraction[10] <= 1
 R92: - 2385.699259840349 Fraction[10] + Q0_var[10] = 0
 R93: Q0_var[10] + Lost_0_expr_10 = 3008.386883879583
 R94: Q0_var[10] + Q_hat_adjusted_10 = 3546.185625082724
 R95: Sold_0[10] - Q0_var[10] - Q1_var[10] + total_stock_second_period_10
   = 0
 R96: total_stock_second_period_10 + Lost_1_expr_10 = 516.2036090416724
 R97: Sold_1[10] + Left_1[10] - total_stock_second_period_10 = 0
 R98: - 700 Sold_0[10] - 700 Sold_1[10] + 700 Lost_0[10] + 700 Lost_1[10]
   + 300 Left_1[10] + profits_vars[10] = 0
 Fraction_lower_bound_11: Fraction[11] >= 0
 Fraction_upper_bound_11: Fraction[11] <= 1
 R101: - 2385.699259840349 Fraction[11] + Q0_var[11] = 0
 R102: Q0_var[11] + Lost_0_expr_11 = 1891.866700850809
 R103: Q0_var[11] + Q_hat_adjusted_11 = 2537.456618024509
 R104: Sold_0[11] - Q0_var[11] - Q1_var[11] + total_stock_second_period_11
   = 0
 R105: total_stock_second_period_11 + Lost_1_expr_11 = 602.857506344179
 R106: Sold_1[11] + Left_1[11] - total_stock_second_period_11 = 0
 R107: - 700 Sold_0[11] - 700 Sold_1[11] + 700 Lost_0[11] + 700 Lost_1[11]
   + 300 Left_1[11] + profits_vars[11] = 0
 Fraction_lower_bound_12: Fraction[12] >= 0
 Fraction_upper_bound_12: Fraction[12] <= 1
 R110: - 2385.699259840349 Fraction[12] + Q0_var[12] = 0
 R111: Q0_var[12] + Lost_0_expr_12 = 980.2110201000349
 R112: Q0_var[12] + Q_hat_adjusted_12 = 1184.56994799468
 R113: Sold_0[12] - Q0_var[12] - Q1_var[12] + total_stock_second_period_12
   = 0
 R114: total_stock_second_period_12 + Lost_1_expr_12 = 139.6023631788834
 R115: Sold_1[12] + Left_1[12] - total_stock_second_period_12 = 0
 R116: - 700 Sold_0[12] - 700 Sold_1[12] + 700 Lost_0[12] + 700 Lost_1[12]
   + 300 Left_1[12] + profits_vars[12] = 0
 Fraction_lower_bound_13: Fraction[13] >= 0
 Fraction_upper_bound_13: Fraction[13] <= 1
 R119: - 2385.699259840349 Fraction[13] + Q0_var[13] = 0
 R120: Q0_var[13] + Lost_0_expr_13 = 1387.732635677946
 R121: Q0_var[13] + Q_hat_adjusted_13 = 1779.316338214647
 R122: Sold_0[13] - Q0_var[13] - Q1_var[13] + total_stock_second_period_13
   = 0
 R123: total_stock_second_period_13 + Lost_1_expr_13 = 349.5940972357466
 R124: Sold_1[13] + Left_1[13] - total_stock_second_period_13 = 0
 R125: - 700 Sold_0[13] - 700 Sold_1[13] + 700 Lost_0[13] + 700 Lost_1[13]
   + 300 Left_1[13] + profits_vars[13] = 0
 Fraction_lower_bound_14: Fraction[14] >= 0
 Fraction_upper_bound_14: Fraction[14] <= 1
 R128: - 2385.699259840349 Fraction[14] + Q0_var[14] = 0
 R129: Q0_var[14] + Lost_0_expr_14 = 2441.242708589496
 R130: Q0_var[14] + Q_hat_adjusted_14 = 2960.090986797536
 R131: Sold_0[14] - Q0_var[14] - Q1_var[14] + total_stock_second_period_14
   = 0
 R132: total_stock_second_period_14 + Lost_1_expr_14 = 525.405888762802
 R133: Sold_1[14] + Left_1[14] - total_stock_second_period_14 = 0
 R134: - 700 Sold_0[14] - 700 Sold_1[14] + 700 Lost_0[14] + 700 Lost_1[14]
   + 300 Left_1[14] + profits_vars[14] = 0
 Fraction_lower_bound_15: Fraction[15] >= 0
 Fraction_upper_bound_15: Fraction[15] <= 1
 R137: - 2385.699259840349 Fraction[15] + Q0_var[15] = 0
 R138: Q0_var[15] + Lost_0_expr_15 = 2012.633819233037
 R139: Q0_var[15] + Q_hat_adjusted_15 = 2339.453873865858
 R140: Sold_0[15] - Q0_var[15] - Q1_var[15] + total_stock_second_period_15
   = 0
 R141: total_stock_second_period_15 + Lost_1_expr_15 = 350.9359618091321
 R142: Sold_1[15] + Left_1[15] - total_stock_second_period_15 = 0
 R143: - 700 Sold_0[15] - 700 Sold_1[15] + 700 Lost_0[15] + 700 Lost_1[15]
   + 300 Left_1[15] + profits_vars[15] = 0
 Fraction_lower_bound_16: Fraction[16] >= 0
 Fraction_upper_bound_16: Fraction[16] <= 1
 R146: - 2385.699259840349 Fraction[16] + Q0_var[16] = 0
 R147: Q0_var[16] + Lost_0_expr_16 = 1378.118594517906
 R148: Q0_var[16] + Q_hat_adjusted_16 = 1473.239000700134
 R149: Sold_0[16] - Q0_var[16] - Q1_var[16] + total_stock_second_period_16
   = 0
 R150: total_stock_second_period_16 + Lost_1_expr_16 = 133.17399695976
 R151: Sold_1[16] + Left_1[16] - total_stock_second_period_16 = 0
 R152: - 700 Sold_0[16] - 700 Sold_1[16] + 700 Lost_0[16] + 700 Lost_1[16]
   + 300 Left_1[16] + profits_vars[16] = 0
 Fraction_lower_bound_17: Fraction[17] >= 0
 Fraction_upper_bound_17: Fraction[17] <= 1
 R155: - 2385.699259840349 Fraction[17] + Q0_var[17] = 0
 R156: Q0_var[17] + Lost_0_expr_17 = 1704.262837628942
 R157: Q0_var[17] + Q_hat_adjusted_17 = 2018.659754040946
 R158: Sold_0[17] - Q0_var[17] - Q1_var[17] + total_stock_second_period_17
   = 0
 R159: total_stock_second_period_17 + Lost_1_expr_17 = 217.6342946536067
 R160: Sold_1[17] + Left_1[17] - total_stock_second_period_17 = 0
 R161: - 700 Sold_0[17] - 700 Sold_1[17] + 700 Lost_0[17] + 700 Lost_1[17]
   + 300 Left_1[17] + profits_vars[17] = 0
 Fraction_lower_bound_18: Fraction[18] >= 0
 Fraction_upper_bound_18: Fraction[18] <= 1
 R164: - 2385.699259840349 Fraction[18] + Q0_var[18] = 0
 R165: Q0_var[18] + Lost_0_expr_18 = 1442.127297952143
 R166: Q0_var[18] + Q_hat_adjusted_18 = 1602.224187247448
 R167: Sold_0[18] - Q0_var[18] - Q1_var[18] + total_stock_second_period_18
   = 0
 R168: total_stock_second_period_18 + Lost_1_expr_18 = 91.62091743269681
 R169: Sold_1[18] + Left_1[18] - total_stock_second_period_18 = 0
 R170: - 700 Sold_0[18] - 700 Sold_1[18] + 700 Lost_0[18] + 700 Lost_1[18]
   + 300 Left_1[18] + profits_vars[18] = 0
 Fraction_lower_bound_19: Fraction[19] >= 0
 Fraction_upper_bound_19: Fraction[19] <= 1
 R173: - 2385.699259840349 Fraction[19] + Q0_var[19] = 0
 R174: Q0_var[19] + Lost_0_expr_19 = 1733.191219036665
 R175: Q0_var[19] + Q_hat_adjusted_19 = 1946.286603185815
 R176: Sold_0[19] - Q0_var[19] - Q1_var[19] + total_stock_second_period_19
   = 0
 R177: total_stock_second_period_19 + Lost_1_expr_19 = 97.47080833391965
 R178: Sold_1[19] + Left_1[19] - total_stock_second_period_19 = 0
 R179: - 700 Sold_0[19] - 700 Sold_1[19] + 700 Lost_0[19] + 700 Lost_1[19]
   + 300 Left_1[19] + profits_vars[19] = 0
 Fraction_lower_bound_20: Fraction[20] >= 0
 Fraction_upper_bound_20: Fraction[20] <= 1
 R182: - 2385.699259840349 Fraction[20] + Q0_var[20] = 0
 R183: Q0_var[20] + Lost_0_expr_20 = 2493.578702885849
 R184: Q0_var[20] + Q_hat_adjusted_20 = 3028.333284008924
 R185: Sold_0[20] - Q0_var[20] - Q1_var[20] + total_stock_second_period_20
   = 0
 R186: total_stock_second_period_20 + Lost_1_expr_20 = 430.602312477249
 R187: Sold_1[20] + Left_1[20] - total_stock_second_period_20 = 0
 R188: - 700 Sold_0[20] - 700 Sold_1[20] + 700 Lost_0[20] + 700 Lost_1[20]
   + 300 Left_1[20] + profits_vars[20] = 0
 Fraction_lower_bound_21: Fraction[21] >= 0
 Fraction_upper_bound_21: Fraction[21] <= 1
 R191: - 2385.699259840349 Fraction[21] + Q0_var[21] = 0
 R192: Q0_var[21] + Lost_0_expr_21 = 1649.037247900663
 R193: Q0_var[21] + Q_hat_adjusted_21 = 2365.836835667496
 R194: Sold_0[21] - Q0_var[21] - Q1_var[21] + total_stock_second_period_21
   = 0
 R195: total_stock_second_period_21 + Lost_1_expr_21 = 742.0163523771482
 R196: Sold_1[21] + Left_1[21] - total_stock_second_period_21 = 0
 R197: - 700 Sold_0[21] - 700 Sold_1[21] + 700 Lost_0[21] + 700 Lost_1[21]
   + 300 Left_1[21] + profits_vars[21] = 0
 Fraction_lower_bound_22: Fraction[22] >= 0
 Fraction_upper_bound_22: Fraction[22] <= 1
 R200: - 2385.699259840349 Fraction[22] + Q0_var[22] = 0
 R201: Q0_var[22] + Lost_0_expr_22 = 599.3036571453245
 R202: Q0_var[22] + Q_hat_adjusted_22 = 1013.594284949571
 R203: Sold_0[22] - Q0_var[22] - Q1_var[22] + total_stock_second_period_22
   = 0
 R204: total_stock_second_period_22 + Lost_1_expr_22 = 444.2067741775776
 R205: Sold_1[22] + Left_1[22] - total_stock_second_period_22 = 0
 R206: - 700 Sold_0[22] - 700 Sold_1[22] + 700 Lost_0[22] + 700 Lost_1[22]
   + 300 Left_1[22] + profits_vars[22] = 0
 Fraction_lower_bound_23: Fraction[23] >= 0
 Fraction_upper_bound_23: Fraction[23] <= 1
 R209: - 2385.699259840349 Fraction[23] + Q0_var[23] = 0
 R210: Q0_var[23] + Lost_0_expr_23 = 1231.597427971065
 R211: Q0_var[23] + Q_hat_adjusted_23 = 1672.491166788958
 R212: Sold_0[23] - Q0_var[23] - Q1_var[23] + total_stock_second_period_23
   = 0
 R213: total_stock_second_period_23 + Lost_1_expr_23 = 359.4178070404722
 R214: Sold_1[23] + Left_1[23] - total_stock_second_period_23 = 0
 R215: - 700 Sold_0[23] - 700 Sold_1[23] + 700 Lost_0[23] + 700 Lost_1[23]
   + 300 Left_1[23] + profits_vars[23] = 0
 Fraction_lower_bound_24: Fraction[24] >= 0
 Fraction_upper_bound_24: Fraction[24] <= 1
 R218: - 2385.699259840349 Fraction[24] + Q0_var[24] = 0
 R219: Q0_var[24] + Lost_0_expr_24 = 1846.022202248738
 R220: Q0_var[24] + Q_hat_adjusted_24 = 2389.033318288848
 R221: Sold_0[24] - Q0_var[24] - Q1_var[24] + total_stock_second_period_24
   = 0
 R222: total_stock_second_period_24 + Lost_1_expr_24 = 551.9228300953447
 R223: Sold_1[24] + Left_1[24] - total_stock_second_period_24 = 0
 R224: - 700 Sold_0[24] - 700 Sold_1[24] + 700 Lost_0[24] + 700 Lost_1[24]
   + 300 Left_1[24] + profits_vars[24] = 0
 Fraction_lower_bound_25: Fraction[25] >= 0
 Fraction_upper_bound_25: Fraction[25] <= 1
 R227: - 2385.699259840349 Fraction[25] + Q0_var[25] = 0
 R228: Q0_var[25] + Lost_0_expr_25 = 2808.674154905328
 R229: Q0_var[25] + Q_hat_adjusted_25 = 3521.913433213405
 R230: Sold_0[25] - Q0_var[25] - Q1_var[25] + total_stock_second_period_25
   = 0
 R231: total_stock_second_period_25 + Lost_1_expr_25 = 741.6546869349409
 R232: Sold_1[25] + Left_1[25] - total_stock_second_period_25 = 0
 R233: - 700 Sold_0[25] - 700 Sold_1[25] + 700 Lost_0[25] + 700 Lost_1[25]
   + 300 Left_1[25] + profits_vars[25] = 0
 Fraction_lower_bound_26: Fraction[26] >= 0
 Fraction_upper_bound_26: Fraction[26] <= 1
 R236: - 2385.699259840349 Fraction[26] + Q0_var[26] = 0
 R237: Q0_var[26] + Lost_0_expr_26 = 1574.018181154822
 R238: Q0_var[26] + Q_hat_adjusted_26 = 1904.627280629476
 R239: Sold_0[26] - Q0_var[26] - Q1_var[26] + total_stock_second_period_26
   = 0
 R240: total_stock_second_period_26 + Lost_1_expr_26 = 151.7074010921853
 R241: Sold_1[26] + Left_1[26] - total_stock_second_period_26 = 0
 R242: - 700 Sold_0[26] - 700 Sold_1[26] + 700 Lost_0[26] + 700 Lost_1[26]
   + 300 Left_1[26] + profits_vars[26] = 0
 Fraction_lower_bound_27: Fraction[27] >= 0
 Fraction_upper_bound_27: Fraction[27] <= 1
 R245: - 2385.699259840349 Fraction[27] + Q0_var[27] = 0
 R246: Q0_var[27] + Lost_0_expr_27 = 756.8146576797355
 R247: Q0_var[27] + Q_hat_adjusted_27 = 1212.294157060115
 R248: Sold_0[27] - Q0_var[27] - Q1_var[27] + total_stock_second_period_27
   = 0
 R249: total_stock_second_period_27 + Lost_1_expr_27 = 339.6570884816433
 R250: Sold_1[27] + Left_1[27] - total_stock_second_period_27 = 0
 R251: - 700 Sold_0[27] - 700 Sold_1[27] + 700 Lost_0[27] + 700 Lost_1[27]
   + 300 Left_1[27] + profits_vars[27] = 0
 Fraction_lower_bound_28: Fraction[28] >= 0
 Fraction_upper_bound_28: Fraction[28] <= 1
 R254: - 2385.699259840349 Fraction[28] + Q0_var[28] = 0
 R255: Q0_var[28] + Lost_0_expr_28 = 1618.529202391011
 R256: Q0_var[28] + Q_hat_adjusted_28 = 2165.361239293313
 R257: Sold_0[28] - Q0_var[28] - Q1_var[28] + total_stock_second_period_28
   = 0
 R258: total_stock_second_period_28 + Lost_1_expr_28 = 551.1150033150486
 R259: Sold_1[28] + Left_1[28] - total_stock_second_period_28 = 0
 R260: - 700 Sold_0[28] - 700 Sold_1[28] + 700 Lost_0[28] + 700 Lost_1[28]
   + 300 Left_1[28] + profits_vars[28] = 0
 Fraction_lower_bound_29: Fraction[29] >= 0
 Fraction_upper_bound_29: Fraction[29] <= 1
 R263: - 2385.699259840349 Fraction[29] + Q0_var[29] = 0
 R264: Q0_var[29] + Lost_0_expr_29 = 311.8630368587254
 R265: Q0_var[29] + Q_hat_adjusted_29 = 684.8883749979473
 R266: Sold_0[29] - Q0_var[29] - Q1_var[29] + total_stock_second_period_29
   = 0
 R267: total_stock_second_period_29 + Lost_1_expr_29 = 442.9415178563536
 R268: Sold_1[29] + Left_1[29] - total_stock_second_period_29 = 0
 R269: - 700 Sold_0[29] - 700 Sold_1[29] + 700 Lost_0[29] + 700 Lost_1[29]
   + 300 Left_1[29] + profits_vars[29] = 0
Bounds
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
 Q0_var[0] <= 2386.699259840349
 Q0_var[1] <= 2386.699259840349
 Q0_var[2] <= 2386.699259840349
 Q0_var[3] <= 2386.699259840349
 Q0_var[4] <= 2386.699259840349
 Q0_var[5] <= 2386.699259840349
 Q0_var[6] <= 2386.699259840349
 Q0_var[7] <= 2386.699259840349
 Q0_var[8] <= 2386.699259840349
 Q0_var[9] <= 2386.699259840349
 Q0_var[10] <= 2386.699259840349
 Q0_var[11] <= 2386.699259840349
 Q0_var[12] <= 2386.699259840349
 Q0_var[13] <= 2386.699259840349
 Q0_var[14] <= 2386.699259840349
 Q0_var[15] <= 2386.699259840349
 Q0_var[16] <= 2386.699259840349
 Q0_var[17] <= 2386.699259840349
 Q0_var[18] <= 2386.699259840349
 Q0_var[19] <= 2386.699259840349
 Q0_var[20] <= 2386.699259840349
 Q0_var[21] <= 2386.699259840349
 Q0_var[22] <= 2386.699259840349
 Q0_var[23] <= 2386.699259840349
 Q0_var[24] <= 2386.699259840349
 Q0_var[25] <= 2386.699259840349
 Q0_var[26] <= 2386.699259840349
 Q0_var[27] <= 2386.699259840349
 Q0_var[28] <= 2386.699259840349
 Q0_var[29] <= 2386.699259840349
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
General Constraints
 min_sales_constr_0: Sold_0[0] = MIN ( Q0_var[0] , 1237.106505172745 )
 max_lost_constr_0: Lost_0[0] = MAX ( Lost_0_expr_0 , 0 )
 max_Q1_constr_0: Q1_var[0] = MAX ( Q_hat_adjusted_0 , 0 )
 min_sales2_constr_0: Sold_1[0] = MIN ( total_stock_second_period_0
   , 307.0146413669897 )
 max_lost2_constr_0: Lost_1[0] = MAX ( Lost_1_expr_0 , 0 )
 min_sales_constr_1: Sold_0[1] = MIN ( Q0_var[1] , 900.4115652482144 )
 max_lost_constr_1: Lost_0[1] = MAX ( Lost_0_expr_1 , 0 )
 max_Q1_constr_1: Q1_var[1] = MAX ( Q_hat_adjusted_1 , 0 )
 min_sales2_constr_1: Sold_1[1] = MIN ( total_stock_second_period_1
   , 19.24180288226607 )
 max_lost2_constr_1: Lost_1[1] = MAX ( Lost_1_expr_1 , 0 )
 min_sales_constr_2: Sold_0[2] = MIN ( Q0_var[2] , 348.3813833935464 )
 max_lost_constr_2: Lost_0[2] = MAX ( Lost_0_expr_2 , 0 )
 max_Q1_constr_2: Q1_var[2] = MAX ( Q_hat_adjusted_2 , 0 )
 min_sales2_constr_2: Sold_1[2] = MIN ( total_stock_second_period_2
   , 169.8599237814332 )
 max_lost2_constr_2: Lost_1[2] = MAX ( Lost_1_expr_2 , 0 )
 min_sales_constr_3: Sold_0[3] = MIN ( Q0_var[3] , 1576.839725251233 )
 max_lost_constr_3: Lost_0[3] = MAX ( Lost_0_expr_3 , 0 )
 max_Q1_constr_3: Q1_var[3] = MAX ( Q_hat_adjusted_3 , 0 )
 min_sales2_constr_3: Sold_1[3] = MIN ( total_stock_second_period_3
   , 765.0737567426161 )
 max_lost2_constr_3: Lost_1[3] = MAX ( Lost_1_expr_3 , 0 )
 min_sales_constr_4: Sold_0[4] = MIN ( Q0_var[4] , 1687.746268160732 )
 max_lost_constr_4: Lost_0[4] = MAX ( Lost_0_expr_4 , 0 )
 max_Q1_constr_4: Q1_var[4] = MAX ( Q_hat_adjusted_4 , 0 )
 min_sales2_constr_4: Sold_1[4] = MIN ( total_stock_second_period_4
   , 410.9698184910955 )
 max_lost2_constr_4: Lost_1[4] = MAX ( Lost_1_expr_4 , 0 )
 min_sales_constr_5: Sold_0[5] = MIN ( Q0_var[5] , 2443.617560988295 )
 max_lost_constr_5: Lost_0[5] = MAX ( Lost_0_expr_5 , 0 )
 max_Q1_constr_5: Q1_var[5] = MAX ( Q_hat_adjusted_5 , 0 )
 min_sales2_constr_5: Sold_1[5] = MIN ( total_stock_second_period_5
   , 622.232022253729 )
 max_lost2_constr_5: Lost_1[5] = MAX ( Lost_1_expr_5 , 0 )
 min_sales_constr_6: Sold_0[6] = MIN ( Q0_var[6] , 2004.656042059087 )
 max_lost_constr_6: Lost_0[6] = MAX ( Lost_0_expr_6 , 0 )
 max_Q1_constr_6: Q1_var[6] = MAX ( Q_hat_adjusted_6 , 0 )
 min_sales2_constr_6: Sold_1[6] = MIN ( total_stock_second_period_6
   , 427.178256319511 )
 max_lost2_constr_6: Lost_1[6] = MAX ( Lost_1_expr_6 , 0 )
 min_sales_constr_7: Sold_0[7] = MIN ( Q0_var[7] , 984.3984055832955 )
 max_lost_constr_7: Lost_0[7] = MAX ( Lost_0_expr_7 , 0 )
 max_Q1_constr_7: Q1_var[7] = MAX ( Q_hat_adjusted_7 , 0 )
 min_sales2_constr_7: Sold_1[7] = MIN ( total_stock_second_period_7
   , 349.8420422049586 )
 max_lost2_constr_7: Lost_1[7] = MAX ( Lost_1_expr_7 , 0 )
 min_sales_constr_8: Sold_0[8] = MIN ( Q0_var[8] , 1302.571392831954 )
 max_lost_constr_8: Lost_0[8] = MAX ( Lost_0_expr_8 , 0 )
 max_Q1_constr_8: Q1_var[8] = MAX ( Q_hat_adjusted_8 , 0 )
 min_sales2_constr_8: Sold_1[8] = MIN ( total_stock_second_period_8
   , 443.5751957473497 )
 max_lost2_constr_8: Lost_1[8] = MAX ( Lost_1_expr_8 , 0 )
 min_sales_constr_9: Sold_0[9] = MIN ( Q0_var[9] , 2170.16072797647 )
 max_lost_constr_9: Lost_0[9] = MAX ( Lost_0_expr_9 , 0 )
 max_Q1_constr_9: Q1_var[9] = MAX ( Q_hat_adjusted_9 , 0 )
 min_sales2_constr_9: Sold_1[9] = MIN ( total_stock_second_period_9
   , 387.8098237156327 )
 max_lost2_constr_9: Lost_1[9] = MAX ( Lost_1_expr_9 , 0 )
 min_sales_constr_10: Sold_0[10] = MIN ( Q0_var[10] , 3008.386883879583 )
 max_lost_constr_10: Lost_0[10] = MAX ( Lost_0_expr_10 , 0 )
 max_Q1_constr_10: Q1_var[10] = MAX ( Q_hat_adjusted_10 , 0 )
 min_sales2_constr_10: Sold_1[10] = MIN ( total_stock_second_period_10
   , 516.2036090416724 )
 max_lost2_constr_10: Lost_1[10] = MAX ( Lost_1_expr_10 , 0 )
 min_sales_constr_11: Sold_0[11] = MIN ( Q0_var[11] , 1891.866700850809 )
 max_lost_constr_11: Lost_0[11] = MAX ( Lost_0_expr_11 , 0 )
 max_Q1_constr_11: Q1_var[11] = MAX ( Q_hat_adjusted_11 , 0 )
 min_sales2_constr_11: Sold_1[11] = MIN ( total_stock_second_period_11
   , 602.857506344179 )
 max_lost2_constr_11: Lost_1[11] = MAX ( Lost_1_expr_11 , 0 )
 min_sales_constr_12: Sold_0[12] = MIN ( Q0_var[12] , 980.2110201000349 )
 max_lost_constr_12: Lost_0[12] = MAX ( Lost_0_expr_12 , 0 )
 max_Q1_constr_12: Q1_var[12] = MAX ( Q_hat_adjusted_12 , 0 )
 min_sales2_constr_12: Sold_1[12] = MIN ( total_stock_second_period_12
   , 139.6023631788834 )
 max_lost2_constr_12: Lost_1[12] = MAX ( Lost_1_expr_12 , 0 )
 min_sales_constr_13: Sold_0[13] = MIN ( Q0_var[13] , 1387.732635677946 )
 max_lost_constr_13: Lost_0[13] = MAX ( Lost_0_expr_13 , 0 )
 max_Q1_constr_13: Q1_var[13] = MAX ( Q_hat_adjusted_13 , 0 )
 min_sales2_constr_13: Sold_1[13] = MIN ( total_stock_second_period_13
   , 349.5940972357466 )
 max_lost2_constr_13: Lost_1[13] = MAX ( Lost_1_expr_13 , 0 )
 min_sales_constr_14: Sold_0[14] = MIN ( Q0_var[14] , 2441.242708589496 )
 max_lost_constr_14: Lost_0[14] = MAX ( Lost_0_expr_14 , 0 )
 max_Q1_constr_14: Q1_var[14] = MAX ( Q_hat_adjusted_14 , 0 )
 min_sales2_constr_14: Sold_1[14] = MIN ( total_stock_second_period_14
   , 525.405888762802 )
 max_lost2_constr_14: Lost_1[14] = MAX ( Lost_1_expr_14 , 0 )
 min_sales_constr_15: Sold_0[15] = MIN ( Q0_var[15] , 2012.633819233037 )
 max_lost_constr_15: Lost_0[15] = MAX ( Lost_0_expr_15 , 0 )
 max_Q1_constr_15: Q1_var[15] = MAX ( Q_hat_adjusted_15 , 0 )
 min_sales2_constr_15: Sold_1[15] = MIN ( total_stock_second_period_15
   , 350.9359618091321 )
 max_lost2_constr_15: Lost_1[15] = MAX ( Lost_1_expr_15 , 0 )
 min_sales_constr_16: Sold_0[16] = MIN ( Q0_var[16] , 1378.118594517906 )
 max_lost_constr_16: Lost_0[16] = MAX ( Lost_0_expr_16 , 0 )
 max_Q1_constr_16: Q1_var[16] = MAX ( Q_hat_adjusted_16 , 0 )
 min_sales2_constr_16: Sold_1[16] = MIN ( total_stock_second_period_16
   , 133.17399695976 )
 max_lost2_constr_16: Lost_1[16] = MAX ( Lost_1_expr_16 , 0 )
 min_sales_constr_17: Sold_0[17] = MIN ( Q0_var[17] , 1704.262837628942 )
 max_lost_constr_17: Lost_0[17] = MAX ( Lost_0_expr_17 , 0 )
 max_Q1_constr_17: Q1_var[17] = MAX ( Q_hat_adjusted_17 , 0 )
 min_sales2_constr_17: Sold_1[17] = MIN ( total_stock_second_period_17
   , 217.6342946536067 )
 max_lost2_constr_17: Lost_1[17] = MAX ( Lost_1_expr_17 , 0 )
 min_sales_constr_18: Sold_0[18] = MIN ( Q0_var[18] , 1442.127297952143 )
 max_lost_constr_18: Lost_0[18] = MAX ( Lost_0_expr_18 , 0 )
 max_Q1_constr_18: Q1_var[18] = MAX ( Q_hat_adjusted_18 , 0 )
 min_sales2_constr_18: Sold_1[18] = MIN ( total_stock_second_period_18
   , 91.62091743269681 )
 max_lost2_constr_18: Lost_1[18] = MAX ( Lost_1_expr_18 , 0 )
 min_sales_constr_19: Sold_0[19] = MIN ( Q0_var[19] , 1733.191219036665 )
 max_lost_constr_19: Lost_0[19] = MAX ( Lost_0_expr_19 , 0 )
 max_Q1_constr_19: Q1_var[19] = MAX ( Q_hat_adjusted_19 , 0 )
 min_sales2_constr_19: Sold_1[19] = MIN ( total_stock_second_period_19
   , 97.47080833391965 )
 max_lost2_constr_19: Lost_1[19] = MAX ( Lost_1_expr_19 , 0 )
 min_sales_constr_20: Sold_0[20] = MIN ( Q0_var[20] , 2493.578702885849 )
 max_lost_constr_20: Lost_0[20] = MAX ( Lost_0_expr_20 , 0 )
 max_Q1_constr_20: Q1_var[20] = MAX ( Q_hat_adjusted_20 , 0 )
 min_sales2_constr_20: Sold_1[20] = MIN ( total_stock_second_period_20
   , 430.602312477249 )
 max_lost2_constr_20: Lost_1[20] = MAX ( Lost_1_expr_20 , 0 )
 min_sales_constr_21: Sold_0[21] = MIN ( Q0_var[21] , 1649.037247900663 )
 max_lost_constr_21: Lost_0[21] = MAX ( Lost_0_expr_21 , 0 )
 max_Q1_constr_21: Q1_var[21] = MAX ( Q_hat_adjusted_21 , 0 )
 min_sales2_constr_21: Sold_1[21] = MIN ( total_stock_second_period_21
   , 742.0163523771482 )
 max_lost2_constr_21: Lost_1[21] = MAX ( Lost_1_expr_21 , 0 )
 min_sales_constr_22: Sold_0[22] = MIN ( Q0_var[22] , 599.3036571453245 )
 max_lost_constr_22: Lost_0[22] = MAX ( Lost_0_expr_22 , 0 )
 max_Q1_constr_22: Q1_var[22] = MAX ( Q_hat_adjusted_22 , 0 )
 min_sales2_constr_22: Sold_1[22] = MIN ( total_stock_second_period_22
   , 444.2067741775776 )
 max_lost2_constr_22: Lost_1[22] = MAX ( Lost_1_expr_22 , 0 )
 min_sales_constr_23: Sold_0[23] = MIN ( Q0_var[23] , 1231.597427971065 )
 max_lost_constr_23: Lost_0[23] = MAX ( Lost_0_expr_23 , 0 )
 max_Q1_constr_23: Q1_var[23] = MAX ( Q_hat_adjusted_23 , 0 )
 min_sales2_constr_23: Sold_1[23] = MIN ( total_stock_second_period_23
   , 359.4178070404722 )
 max_lost2_constr_23: Lost_1[23] = MAX ( Lost_1_expr_23 , 0 )
 min_sales_constr_24: Sold_0[24] = MIN ( Q0_var[24] , 1846.022202248738 )
 max_lost_constr_24: Lost_0[24] = MAX ( Lost_0_expr_24 , 0 )
 max_Q1_constr_24: Q1_var[24] = MAX ( Q_hat_adjusted_24 , 0 )
 min_sales2_constr_24: Sold_1[24] = MIN ( total_stock_second_period_24
   , 551.9228300953447 )
 max_lost2_constr_24: Lost_1[24] = MAX ( Lost_1_expr_24 , 0 )
 min_sales_constr_25: Sold_0[25] = MIN ( Q0_var[25] , 2808.674154905328 )
 max_lost_constr_25: Lost_0[25] = MAX ( Lost_0_expr_25 , 0 )
 max_Q1_constr_25: Q1_var[25] = MAX ( Q_hat_adjusted_25 , 0 )
 min_sales2_constr_25: Sold_1[25] = MIN ( total_stock_second_period_25
   , 741.6546869349409 )
 max_lost2_constr_25: Lost_1[25] = MAX ( Lost_1_expr_25 , 0 )
 min_sales_constr_26: Sold_0[26] = MIN ( Q0_var[26] , 1574.018181154822 )
 max_lost_constr_26: Lost_0[26] = MAX ( Lost_0_expr_26 , 0 )
 max_Q1_constr_26: Q1_var[26] = MAX ( Q_hat_adjusted_26 , 0 )
 min_sales2_constr_26: Sold_1[26] = MIN ( total_stock_second_period_26
   , 151.7074010921853 )
 max_lost2_constr_26: Lost_1[26] = MAX ( Lost_1_expr_26 , 0 )
 min_sales_constr_27: Sold_0[27] = MIN ( Q0_var[27] , 756.8146576797355 )
 max_lost_constr_27: Lost_0[27] = MAX ( Lost_0_expr_27 , 0 )
 max_Q1_constr_27: Q1_var[27] = MAX ( Q_hat_adjusted_27 , 0 )
 min_sales2_constr_27: Sold_1[27] = MIN ( total_stock_second_period_27
   , 339.6570884816433 )
 max_lost2_constr_27: Lost_1[27] = MAX ( Lost_1_expr_27 , 0 )
 min_sales_constr_28: Sold_0[28] = MIN ( Q0_var[28] , 1618.529202391011 )
 max_lost_constr_28: Lost_0[28] = MAX ( Lost_0_expr_28 , 0 )
 max_Q1_constr_28: Q1_var[28] = MAX ( Q_hat_adjusted_28 , 0 )
 min_sales2_constr_28: Sold_1[28] = MIN ( total_stock_second_period_28
   , 551.1150033150486 )
 max_lost2_constr_28: Lost_1[28] = MAX ( Lost_1_expr_28 , 0 )
 min_sales_constr_29: Sold_0[29] = MIN ( Q0_var[29] , 311.8630368587254 )
 max_lost_constr_29: Lost_0[29] = MAX ( Lost_0_expr_29 , 0 )
 max_Q1_constr_29: Q1_var[29] = MAX ( Q_hat_adjusted_29 , 0 )
 min_sales2_constr_29: Sold_1[29] = MIN ( total_stock_second_period_29
   , 442.9415178563536 )
 max_lost2_constr_29: Lost_1[29] = MAX ( Lost_1_expr_29 , 0 )
End
