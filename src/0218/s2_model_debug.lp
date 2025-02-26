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
 R2: - 0.6455745747688753 alphas[0] - 0.7580687118380918 alphas[1]
   - 0.3487370079969519 alphas[2] + 1.097429753547478 alphas[3] - alphas[4]
   + f_var[0] = 0
 R3: - 2433.451666584996 Fraction[0] + Q0_var[0] = 0
 R4: Q0_var[0] + Lost_0_expr_0 = 2592.957502623658
 R5: Q0_var[0] + Q_hat_adjusted_0 = 3171.383301080468
 R6: Sold_0[0] - Q0_var[0] - Q1_var[0] + total_stock_second_period_0 = 0
 R7: total_stock_second_period_0 + Lost_1_expr_0 = 468.3804544210274
 R8: Sold_1[0] + Left_1[0] - total_stock_second_period_0 = 0
 R9: - 700 Sold_0[0] - 700 Sold_1[0] + 700 Lost_0[0] + 700 Lost_1[0]
   + 300 Left_1[0] + profits_vars[0] = 0
 Fraction_lower_bound_1: Fraction[1] >= 0
 Fraction_upper_bound_1: Fraction[1] <= 1
 R12: 0.2500162638730395 alphas[0] - 1.320429564616989 alphas[1]
   + 1.611633822204492 alphas[2] - 0.4211015375937367 alphas[3] - alphas[4]
   + f_var[1] = 0
 R13: - 2433.451666584996 Fraction[1] + Q0_var[1] = 0
 R14: Q0_var[1] + Lost_0_expr_1 = 527.6091202858785
 R15: Q0_var[1] + Q_hat_adjusted_1 = 825.6302753248074
 R16: Sold_0[1] - Q0_var[1] - Q1_var[1] + total_stock_second_period_1 = 0
 R17: total_stock_second_period_1 + Lost_1_expr_1 = 268.6975352627283
 R18: Sold_1[1] + Left_1[1] - total_stock_second_period_1 = 0
 R19: - 700 Sold_0[1] - 700 Sold_1[1] + 700 Lost_0[1] + 700 Lost_1[1]
   + 300 Left_1[1] + profits_vars[1] = 0
 Fraction_lower_bound_2: Fraction[2] >= 0
 Fraction_upper_bound_2: Fraction[2] <= 1
 R22: - 1.568759274473915 alphas[0] + 1.546619575883949 alphas[1]
   + 1.689276499138305 alphas[2] + 0.723421055394498 alphas[3] - alphas[4]
   + f_var[2] = 0
 R23: - 2433.451666584996 Fraction[2] + Q0_var[2] = 0
 R24: Q0_var[2] + Lost_0_expr_2 = 1253.414429582754
 R25: Q0_var[2] + Q_hat_adjusted_2 = 1320.824605290604
 R26: Sold_0[2] - Q0_var[2] - Q1_var[2] + total_stock_second_period_2 = 0
 R27: total_stock_second_period_2 + Lost_1_expr_2 = 61.85760523940101
 R28: Sold_1[2] + Left_1[2] - total_stock_second_period_2 = 0
 R29: - 700 Sold_0[2] - 700 Sold_1[2] + 700 Lost_0[2] + 700 Lost_1[2]
   + 300 Left_1[2] + profits_vars[2] = 0
 Fraction_lower_bound_3: Fraction[3] >= 0
 Fraction_upper_bound_3: Fraction[3] <= 1
 R32: - 0.3205649768014743 alphas[0] - 1.219822899076555 alphas[1]
   + 0.5766252125020424 alphas[2] + 1.243015148204178 alphas[3] - alphas[4]
   + f_var[3] = 0
 R33: - 2433.451666584996 Fraction[3] + Q0_var[3] = 0
 R34: Q0_var[3] + Lost_0_expr_3 = 927.3474209617464
 R35: Q0_var[3] + Q_hat_adjusted_3 = 1615.77555094277
 R36: Sold_0[3] - Q0_var[3] - Q1_var[3] + total_stock_second_period_3 = 0
 R37: total_stock_second_period_3 + Lost_1_expr_3 = 762.717631850279
 R38: Sold_1[3] + Left_1[3] - total_stock_second_period_3 = 0
 R39: - 700 Sold_0[3] - 700 Sold_1[3] + 700 Lost_0[3] + 700 Lost_1[3]
   + 300 Left_1[3] + profits_vars[3] = 0
 Fraction_lower_bound_4: Fraction[4] >= 0
 Fraction_upper_bound_4: Fraction[4] <= 1
 R42: 0.5598490548195897 alphas[0] + 0.386035237834075 alphas[1]
   - 1.057020342872136 alphas[2] - 1.455107623256937 alphas[3] - alphas[4]
   + f_var[4] = 0
 R43: - 2433.451666584996 Fraction[4] + Q0_var[4] = 0
 R44: Q0_var[4] + Lost_0_expr_4 = 2258.646338916769
 R45: Q0_var[4] + Q_hat_adjusted_4 = 2895.729062364376
 R46: Sold_0[4] - Q0_var[4] - Q1_var[4] + total_stock_second_period_4 = 0
 R47: total_stock_second_period_4 + Lost_1_expr_4 = 543.5597790620841
 R48: Sold_1[4] + Left_1[4] - total_stock_second_period_4 = 0
 R49: - 700 Sold_0[4] - 700 Sold_1[4] + 700 Lost_0[4] + 700 Lost_1[4]
   + 300 Left_1[4] + profits_vars[4] = 0
 Fraction_lower_bound_5: Fraction[5] >= 0
 Fraction_upper_bound_5: Fraction[5] <= 1
 R52: - 0.1701986535258331 alphas[0] - 1.164545051129923 alphas[1]
   - 1.20180816033887 alphas[2] - 0.3076769443064232 alphas[3] - alphas[4]
   + f_var[5] = 0
 R53: - 2433.451666584996 Fraction[5] + Q0_var[5] = 0
 R54: Q0_var[5] + Lost_0_expr_5 = 2895.569709255679
 R55: Q0_var[5] + Q_hat_adjusted_5 = 3566.462118562437
 R56: Sold_0[5] - Q0_var[5] - Q1_var[5] + total_stock_second_period_5 = 0
 R57: total_stock_second_period_5 + Lost_1_expr_5 = 641.6202046038665
 R58: Sold_1[5] + Left_1[5] - total_stock_second_period_5 = 0
 R59: - 700 Sold_0[5] - 700 Sold_1[5] + 700 Lost_0[5] + 700 Lost_1[5]
   + 300 Left_1[5] + profits_vars[5] = 0
 Fraction_lower_bound_6: Fraction[6] >= 0
 Fraction_upper_bound_6: Fraction[6] <= 1
 R62: 2.17826378746964 alphas[0] - 0.5658056513731251 alphas[1]
   - 1.523121696105617 alphas[2] + 0.9986133260033201 alphas[3] - alphas[4]
   + f_var[6] = 0
 R63: - 2433.451666584996 Fraction[6] + Q0_var[6] = 0
 R64: Q0_var[6] + Lost_0_expr_6 = 2267.079241103794
 R65: Q0_var[6] + Q_hat_adjusted_6 = 2847.590827945382
 R66: Sold_0[6] - Q0_var[6] - Q1_var[6] + total_stock_second_period_6 = 0
 R67: total_stock_second_period_6 + Lost_1_expr_6 = 665.980137395507
 R68: Sold_1[6] + Left_1[6] - total_stock_second_period_6 = 0
 R69: - 700 Sold_0[6] - 700 Sold_1[6] + 700 Lost_0[6] + 700 Lost_1[6]
   + 300 Left_1[6] + profits_vars[6] = 0
 Fraction_lower_bound_7: Fraction[7] >= 0
 Fraction_upper_bound_7: Fraction[7] <= 1
 R72: 1.048899674912968 alphas[0] - 1.512373692495633 alphas[1]
   + 1.516636231760764 alphas[2] - 1.205912418839491 alphas[3] - alphas[4]
   + f_var[7] = 0
 R73: - 2433.451666584996 Fraction[7] + Q0_var[7] = 0
 R74: Q0_var[7] + Lost_0_expr_7 = 1693.557791064872
 R75: Q0_var[7] + Q_hat_adjusted_7 = 1772.835379363624
 R76: Sold_0[7] - Q0_var[7] - Q1_var[7] + total_stock_second_period_7 = 0
 R77: total_stock_second_period_7 + Lost_1_expr_7 = 25.63723783253555
 R78: Sold_1[7] + Left_1[7] - total_stock_second_period_7 = 0
 R79: - 700 Sold_0[7] - 700 Sold_1[7] + 700 Lost_0[7] + 700 Lost_1[7]
   + 300 Left_1[7] + profits_vars[7] = 0
 Fraction_lower_bound_8: Fraction[8] >= 0
 Fraction_upper_bound_8: Fraction[8] <= 1
 R82: - 0.3852890249228113 alphas[0] - 0.78443540666514 alphas[1]
   + 1.586379154390703 alphas[2] - 1.617991846713082 alphas[3] - alphas[4]
   + f_var[8] = 0
 R83: - 2433.451666584996 Fraction[8] + Q0_var[8] = 0
 R84: Q0_var[8] + Lost_0_expr_8 = 1027.357280852033
 R85: Q0_var[8] + Q_hat_adjusted_8 = 1348.220629529618
 R86: Sold_0[8] - Q0_var[8] - Q1_var[8] + total_stock_second_period_8 = 0
 R87: total_stock_second_period_8 + Lost_1_expr_8 = 280.6622024003613
 R88: Sold_1[8] + Left_1[8] - total_stock_second_period_8 = 0
 R89: - 700 Sold_0[8] - 700 Sold_1[8] + 700 Lost_0[8] + 700 Lost_1[8]
   + 300 Left_1[8] + profits_vars[8] = 0
 Fraction_lower_bound_9: Fraction[9] >= 0
 Fraction_upper_bound_9: Fraction[9] <= 1
 R92: 1.094881902221099 alphas[0] + 0.4754822990951114 alphas[1]
   - 1.361820042396026 alphas[2] - 0.0475900502941662 alphas[3] - alphas[4]
   + f_var[9] = 0
 R93: - 2433.451666584996 Fraction[9] + Q0_var[9] = 0
 R94: Q0_var[9] + Lost_0_expr_9 = 2172.665078693703
 R95: Q0_var[9] + Q_hat_adjusted_9 = 2580.012281175921
 R96: Sold_0[9] - Q0_var[9] - Q1_var[9] + total_stock_second_period_9 = 0
 R97: total_stock_second_period_9 + Lost_1_expr_9 = 371.1052747784937
 R98: Sold_1[9] + Left_1[9] - total_stock_second_period_9 = 0
 R99: - 700 Sold_0[9] - 700 Sold_1[9] + 700 Lost_0[9] + 700 Lost_1[9]
   + 300 Left_1[9] + profits_vars[9] = 0
 Fraction_lower_bound_10: Fraction[10] >= 0
 Fraction_upper_bound_10: Fraction[10] <= 1
 R102: - 0.2166808305868786 alphas[0] + 0.0397446445844663 alphas[1]
   + 0.7023882708629388 alphas[2] + 1.145367397323444 alphas[3] - alphas[4]
   + f_var[10] = 0
 R103: - 2433.451666584996 Fraction[10] + Q0_var[10] = 0
 R104: Q0_var[10] + Lost_0_expr_10 = 1493.506928874833
 R105: Q0_var[10] + Q_hat_adjusted_10 = 1965.99019186423
 R106: Sold_0[10] - Q0_var[10] - Q1_var[10] + total_stock_second_period_10
   = 0
 R107: total_stock_second_period_10 + Lost_1_expr_10 = 316.4556753408243
 R108: Sold_1[10] + Left_1[10] - total_stock_second_period_10 = 0
 R109: - 700 Sold_0[10] - 700 Sold_1[10] + 700 Lost_0[10] + 700 Lost_1[10]
   + 300 Left_1[10] + profits_vars[10] = 0
 Fraction_lower_bound_11: Fraction[11] >= 0
 Fraction_upper_bound_11: Fraction[11] <= 1
 R112: 0.5401972704710009 alphas[0] + 0.4376771254328439 alphas[1]
   - 0.1560310424878339 alphas[2] + 0.9248882696691113 alphas[3]
   - alphas[4] + f_var[11] = 0
 R113: - 2433.451666584996 Fraction[11] + Q0_var[11] = 0
 R114: Q0_var[11] + Lost_0_expr_11 = 1592.663883481324
 R115: Q0_var[11] + Q_hat_adjusted_11 = 2066.853695014152
 R116: Sold_0[11] - Q0_var[11] - Q1_var[11] + total_stock_second_period_11
   = 0
 R117: total_stock_second_period_11 + Lost_1_expr_11 = 313.9071128191609
 R118: Sold_1[11] + Left_1[11] - total_stock_second_period_11 = 0
 R119: - 700 Sold_0[11] - 700 Sold_1[11] + 700 Lost_0[11] + 700 Lost_1[11]
   + 300 Left_1[11] + profits_vars[11] = 0
 Fraction_lower_bound_12: Fraction[12] >= 0
 Fraction_upper_bound_12: Fraction[12] <= 1
 R122: 1.713207181804032 alphas[0] + 0.4743102441730999 alphas[1]
   - 1.596536652475708 alphas[2] + 0.1490675401484341 alphas[3] - alphas[4]
   + f_var[12] = 0
 R123: - 2433.451666584996 Fraction[12] + Q0_var[12] = 0
 R124: Q0_var[12] + Lost_0_expr_12 = 2230.347415439984
 R125: Q0_var[12] + Q_hat_adjusted_12 = 2880.039311205012
 R126: Sold_0[12] - Q0_var[12] - Q1_var[12] + total_stock_second_period_12
   = 0
 R127: total_stock_second_period_12 + Lost_1_expr_12 = 622.1450259213642
 R128: Sold_1[12] + Left_1[12] - total_stock_second_period_12 = 0
 R129: - 700 Sold_0[12] - 700 Sold_1[12] + 700 Lost_0[12] + 700 Lost_1[12]
   + 300 Left_1[12] + profits_vars[12] = 0
 Fraction_lower_bound_13: Fraction[13] >= 0
 Fraction_upper_bound_13: Fraction[13] <= 1
 R132: 1.062067744704293 alphas[0] + 0.8062791522019904 alphas[1]
   + 0.3786176078406262 alphas[2] - 1.331055052705376 alphas[3] - alphas[4]
   + f_var[13] = 0
 R133: - 2433.451666584996 Fraction[13] + Q0_var[13] = 0
 R134: Q0_var[13] + Lost_0_expr_13 = 1481.786657512529
 R135: Q0_var[13] + Q_hat_adjusted_13 = 1934.611479943215
 R136: Sold_0[13] - Q0_var[13] - Q1_var[13] + total_stock_second_period_13
   = 0
 R137: total_stock_second_period_13 + Lost_1_expr_13 = 399.05533900029
 R138: Sold_1[13] + Left_1[13] - total_stock_second_period_13 = 0
 R139: - 700 Sold_0[13] - 700 Sold_1[13] + 700 Lost_0[13] + 700 Lost_1[13]
   + 300 Left_1[13] + profits_vars[13] = 0
 Fraction_lower_bound_14: Fraction[14] >= 0
 Fraction_upper_bound_14: Fraction[14] <= 1
 R142: - 0.0245063849937326 alphas[0] + 0.0560000071787025 alphas[1]
   - 0.5221631295925711 alphas[2] + 0.8353153826984892 alphas[3]
   - alphas[4] + f_var[14] = 0
 R143: - 2433.451666584996 Fraction[14] + Q0_var[14] = 0
 R144: Q0_var[14] + Lost_0_expr_14 = 2020.616520896928
 R145: Q0_var[14] + Q_hat_adjusted_14 = 2309.594746567136
 R146: Sold_0[14] - Q0_var[14] - Q1_var[14] + total_stock_second_period_14
   = 0
 R147: total_stock_second_period_14 + Lost_1_expr_14 = 293.1924253825994
 R148: Sold_1[14] + Left_1[14] - total_stock_second_period_14 = 0
 R149: - 700 Sold_0[14] - 700 Sold_1[14] + 700 Lost_0[14] + 700 Lost_1[14]
   + 300 Left_1[14] + profits_vars[14] = 0
 Fraction_lower_bound_15: Fraction[15] >= 0
 Fraction_upper_bound_15: Fraction[15] <= 1
 R152: - 0.1081257657986034 alphas[0] - 0.5369365081346937 alphas[1]
   + 0.377154800448596 alphas[2] - 1.150302262817461 alphas[3] - alphas[4]
   + f_var[15] = 0
 R153: - 2433.451666584996 Fraction[15] + Q0_var[15] = 0
 R154: Q0_var[15] + Lost_0_expr_15 = 1602.318516547827
 R155: Q0_var[15] + Q_hat_adjusted_15 = 2128.811319637733
 R156: Sold_0[15] - Q0_var[15] - Q1_var[15] + total_stock_second_period_15
   = 0
 R157: total_stock_second_period_15 + Lost_1_expr_15 = 554.1257998625251
 R158: Sold_1[15] + Left_1[15] - total_stock_second_period_15 = 0
 R159: - 700 Sold_0[15] - 700 Sold_1[15] + 700 Lost_0[15] + 700 Lost_1[15]
   + 300 Left_1[15] + profits_vars[15] = 0
 Fraction_lower_bound_16: Fraction[16] >= 0
 Fraction_upper_bound_16: Fraction[16] <= 1
 R162: - 0.0419452305242105 alphas[0] + 0.7538197254486717 alphas[1]
   - 0.8694234443248408 alphas[2] + 1.226188279008046 alphas[3] - alphas[4]
   + f_var[16] = 0
 R163: - 2433.451666584996 Fraction[16] + Q0_var[16] = 0
 R164: Q0_var[16] + Lost_0_expr_16 = 1633.236286342205
 R165: Q0_var[16] + Q_hat_adjusted_16 = 2433.938032500053
 R166: Sold_0[16] - Q0_var[16] - Q1_var[16] + total_stock_second_period_16
   = 0
 R167: total_stock_second_period_16 + Lost_1_expr_16 = 810.1495778715318
 R168: Sold_1[16] + Left_1[16] - total_stock_second_period_16 = 0
 R169: - 700 Sold_0[16] - 700 Sold_1[16] + 700 Lost_0[16] + 700 Lost_1[16]
   + 300 Left_1[16] + profits_vars[16] = 0
 Fraction_lower_bound_17: Fraction[17] >= 0
 Fraction_upper_bound_17: Fraction[17] <= 1
 R172: - 0.3574021634827085 alphas[0] - 0.03467701821645 alphas[1]
   - 0.0532929355421212 alphas[2] - 0.0734050430525894 alphas[3]
   - alphas[4] + f_var[17] = 0
 R173: - 2433.451666584996 Fraction[17] + Q0_var[17] = 0
 R174: Q0_var[17] + Lost_0_expr_17 = 1764.326703923937
 R175: Q0_var[17] + Q_hat_adjusted_17 = 2379.436960778419
 R176: Sold_0[17] - Q0_var[17] - Q1_var[17] + total_stock_second_period_17
   = 0
 R177: total_stock_second_period_17 + Lost_1_expr_17 = 496.405442128
 R178: Sold_1[17] + Left_1[17] - total_stock_second_period_17 = 0
 R179: - 700 Sold_0[17] - 700 Sold_1[17] + 700 Lost_0[17] + 700 Lost_1[17]
   + 300 Left_1[17] + profits_vars[17] = 0
 Fraction_lower_bound_18: Fraction[18] >= 0
 Fraction_upper_bound_18: Fraction[18] <= 1
 R182: - 0.3530125951189564 alphas[0] + 1.264377351988401 alphas[1]
   + 0.9315468357698041 alphas[2] - 0.6227452094222327 alphas[3]
   - alphas[4] + f_var[18] = 0
 R183: - 2433.451666584996 Fraction[18] + Q0_var[18] = 0
 R184: Q0_var[18] + Lost_0_expr_18 = 1255.703818018699
 R185: Q0_var[18] + Q_hat_adjusted_18 = 1578.464968101734
 R186: Sold_0[18] - Q0_var[18] - Q1_var[18] + total_stock_second_period_18
   = 0
 R187: total_stock_second_period_18 + Lost_1_expr_18 = 270.7629070749643
 R188: Sold_1[18] + Left_1[18] - total_stock_second_period_18 = 0
 R189: - 700 Sold_0[18] - 700 Sold_1[18] + 700 Lost_0[18] + 700 Lost_1[18]
   + 300 Left_1[18] + profits_vars[18] = 0
 Fraction_lower_bound_19: Fraction[19] >= 0
 Fraction_upper_bound_19: Fraction[19] <= 1
 R192: 0.5295998560354932 alphas[0] + 1.128838286129897 alphas[1]
   + 0.8168956630743474 alphas[2] + 1.092815618756865 alphas[3] - alphas[4]
   + f_var[19] = 0
 R193: - 2433.451666584996 Fraction[19] + Q0_var[19] = 0
 R194: Q0_var[19] + Lost_0_expr_19 = 1038.6059163213
 R195: Q0_var[19] + Q_hat_adjusted_19 = 1529.986393312062
 R196: Sold_0[19] - Q0_var[19] - Q1_var[19] + total_stock_second_period_19
   = 0
 R197: total_stock_second_period_19 + Lost_1_expr_19 = 418.2221889975829
 R198: Sold_1[19] + Left_1[19] - total_stock_second_period_19 = 0
 R199: - 700 Sold_0[19] - 700 Sold_1[19] + 700 Lost_0[19] + 700 Lost_1[19]
   + 300 Left_1[19] + profits_vars[19] = 0
 Fraction_lower_bound_20: Fraction[20] >= 0
 Fraction_upper_bound_20: Fraction[20] <= 1
 R202: - 1.330645350984932 alphas[0] + 1.489634092273405 alphas[1]
   + 0.8618523862039176 alphas[2] - 0.5356937334142946 alphas[3]
   - alphas[4] + f_var[20] = 0
 R203: - 2433.451666584996 Fraction[20] + Q0_var[20] = 0
 R204: Q0_var[20] + Lost_0_expr_20 = 518.7184646007953
 R205: Q0_var[20] + Q_hat_adjusted_20 = 1156.578169968371
 R206: Sold_0[20] - Q0_var[20] - Q1_var[20] + total_stock_second_period_20
   = 0
 R207: total_stock_second_period_20 + Lost_1_expr_20 = 566.9280970338278
 R208: Sold_1[20] + Left_1[20] - total_stock_second_period_20 = 0
 R209: - 700 Sold_0[20] - 700 Sold_1[20] + 700 Lost_0[20] + 700 Lost_1[20]
   + 300 Left_1[20] + profits_vars[20] = 0
 Fraction_lower_bound_21: Fraction[21] >= 0
 Fraction_upper_bound_21: Fraction[21] <= 1
 R212: 0.7849894019828235 alphas[0] - 1.459334821759661 alphas[1]
   - 0.0689232325243904 alphas[2] - 1.504020314880621 alphas[3] - alphas[4]
   + f_var[21] = 0
 R213: - 2433.451666584996 Fraction[21] + Q0_var[21] = 0
 R214: Q0_var[21] + Lost_0_expr_21 = 1468.173150442183
 R215: Q0_var[21] + Q_hat_adjusted_21 = 1846.413056339083
 R216: Sold_0[21] - Q0_var[21] - Q1_var[21] + total_stock_second_period_21
   = 0
 R217: total_stock_second_period_21 + Lost_1_expr_21 = 432.0364270309086
 R218: Sold_1[21] + Left_1[21] - total_stock_second_period_21 = 0
 R219: - 700 Sold_0[21] - 700 Sold_1[21] + 700 Lost_0[21] + 700 Lost_1[21]
   + 300 Left_1[21] + profits_vars[21] = 0
 Fraction_lower_bound_22: Fraction[22] >= 0
 Fraction_upper_bound_22: Fraction[22] <= 1
 R222: 0.5118164533062762 alphas[0] + 1.626708439600186 alphas[1]
   - 0.6488100609973382 alphas[2] + 0.7994153002801776 alphas[3]
   - alphas[4] + f_var[22] = 0
 R223: - 2433.451666584996 Fraction[22] + Q0_var[22] = 0
 R224: Q0_var[22] + Lost_0_expr_22 = 2362.440572219673
 R225: Q0_var[22] + Q_hat_adjusted_22 = 2661.478302455944
 R226: Sold_0[22] - Q0_var[22] - Q1_var[22] + total_stock_second_period_22
   = 0
 R227: total_stock_second_period_22 + Lost_1_expr_22 = 260.9246159553333
 R228: Sold_1[22] + Left_1[22] - total_stock_second_period_22 = 0
 R229: - 700 Sold_0[22] - 700 Sold_1[22] + 700 Lost_0[22] + 700 Lost_1[22]
   + 300 Left_1[22] + profits_vars[22] = 0
 Fraction_lower_bound_23: Fraction[23] >= 0
 Fraction_upper_bound_23: Fraction[23] <= 1
 R232: - 1.312155045867253 alphas[0] + 1.069041238514881 alphas[1]
   - 1.483982754882942 alphas[2] - 0.8730633521864885 alphas[3] - alphas[4]
   + f_var[23] = 0
 R233: - 2433.451666584996 Fraction[23] + Q0_var[23] = 0
 R234: Q0_var[23] + Lost_0_expr_23 = 2000.47567035108
 R235: Q0_var[23] + Q_hat_adjusted_23 = 2410.294944791175
 R236: Sold_0[23] - Q0_var[23] - Q1_var[23] + total_stock_second_period_23
   = 0
 R237: total_stock_second_period_23 + Lost_1_expr_23 = 528.5532123430869
 R238: Sold_1[23] + Left_1[23] - total_stock_second_period_23 = 0
 R239: - 700 Sold_0[23] - 700 Sold_1[23] + 700 Lost_0[23] + 700 Lost_1[23]
   + 300 Left_1[23] + profits_vars[23] = 0
 Fraction_lower_bound_24: Fraction[24] >= 0
 Fraction_upper_bound_24: Fraction[24] <= 1
 R242: - 0.9692868737857838 alphas[0] - 0.317108185792482 alphas[1]
   - 0.6179213021285588 alphas[2] - 0.5190413917636644 alphas[3]
   - alphas[4] + f_var[24] = 0
 R243: - 2433.451666584996 Fraction[24] + Q0_var[24] = 0
 R244: Q0_var[24] + Lost_0_expr_24 = 2455.346290371662
 R245: Q0_var[24] + Q_hat_adjusted_24 = 2988.556054986683
 R246: Sold_0[24] - Q0_var[24] - Q1_var[24] + total_stock_second_period_24
   = 0
 R247: total_stock_second_period_24 + Lost_1_expr_24 = 301.2664997991592
 R248: Sold_1[24] + Left_1[24] - total_stock_second_period_24 = 0
 R249: - 700 Sold_0[24] - 700 Sold_1[24] + 700 Lost_0[24] + 700 Lost_1[24]
   + 300 Left_1[24] + profits_vars[24] = 0
 Fraction_lower_bound_25: Fraction[25] >= 0
 Fraction_upper_bound_25: Fraction[25] <= 1
 R252: - 0.5601240742820119 alphas[0] + 1.379315697570564 alphas[1]
   - 0.913536108762612 alphas[2] + 0.7263990422098 alphas[3] - alphas[4]
   + f_var[25] = 0
 R253: - 2433.451666584996 Fraction[25] + Q0_var[25] = 0
 R254: Q0_var[25] + Lost_0_expr_25 = 2144.833376865706
 R255: Q0_var[25] + Q_hat_adjusted_25 = 2725.249996362933
 R256: Sold_0[25] - Q0_var[25] - Q1_var[25] + total_stock_second_period_25
   = 0
 R257: total_stock_second_period_25 + Lost_1_expr_25 = 652.8604790557135
 R258: Sold_1[25] + Left_1[25] - total_stock_second_period_25 = 0
 R259: - 700 Sold_0[25] - 700 Sold_1[25] + 700 Lost_0[25] + 700 Lost_1[25]
   + 300 Left_1[25] + profits_vars[25] = 0
 Fraction_lower_bound_26: Fraction[26] >= 0
 Fraction_upper_bound_26: Fraction[26] <= 1
 R262: 1.854175623382176 alphas[0] - 1.175792921053671 alphas[1]
   + 1.164348469135402 alphas[2] + 1.251958647211864 alphas[3] - alphas[4]
   + f_var[26] = 0
 R263: - 2433.451666584996 Fraction[26] + Q0_var[26] = 0
 R264: Q0_var[26] + Lost_0_expr_26 = 825.7637533755192
 R265: Q0_var[26] + Q_hat_adjusted_26 = 1175.145200976889
 R266: Sold_0[26] - Q0_var[26] - Q1_var[26] + total_stock_second_period_26
   = 0
 R267: total_stock_second_period_26 + Lost_1_expr_26 = 322.0635548971552
 R268: Sold_1[26] + Left_1[26] - total_stock_second_period_26 = 0
 R269: - 700 Sold_0[26] - 700 Sold_1[26] + 700 Lost_0[26] + 700 Lost_1[26]
   + 300 Left_1[26] + profits_vars[26] = 0
 Fraction_lower_bound_27: Fraction[27] >= 0
 Fraction_upper_bound_27: Fraction[27] <= 1
 R272: - 1.422376235041627 alphas[0] - 0.6578419488117777 alphas[1]
   + 0.468164977284001 alphas[2] + 1.242908057316128 alphas[3] - alphas[4]
   + f_var[27] = 0
 R273: - 2433.451666584996 Fraction[27] + Q0_var[27] = 0
 R274: Q0_var[27] + Lost_0_expr_27 = 1340.999558839349
 R275: Q0_var[27] + Q_hat_adjusted_27 = 1757.976888243217
 R276: Sold_0[27] - Q0_var[27] - Q1_var[27] + total_stock_second_period_27
   = 0
 R277: total_stock_second_period_27 + Lost_1_expr_27 = 286.8666011770293
 R278: Sold_1[27] + Left_1[27] - total_stock_second_period_27 = 0
 R279: - 700 Sold_0[27] - 700 Sold_1[27] + 700 Lost_0[27] + 700 Lost_1[27]
   + 300 Left_1[27] + profits_vars[27] = 0
 Fraction_lower_bound_28: Fraction[28] >= 0
 Fraction_upper_bound_28: Fraction[28] <= 1
 R282: - 0.6015289858178829 alphas[0] - 1.445477784899326 alphas[1]
   + 0.3333731063156596 alphas[2] - 0.2928483377881266 alphas[3]
   - alphas[4] + f_var[28] = 0
 R283: - 2433.451666584996 Fraction[28] + Q0_var[28] = 0
 R284: Q0_var[28] + Lost_0_expr_28 = 602.5924612900915
 R285: Q0_var[28] + Q_hat_adjusted_28 = 1018.291338116053
 R286: Sold_0[28] - Q0_var[28] - Q1_var[28] + total_stock_second_period_28
   = 0
 R287: total_stock_second_period_28 + Lost_1_expr_28 = 318.6459754579744
 R288: Sold_1[28] + Left_1[28] - total_stock_second_period_28 = 0
 R289: - 700 Sold_0[28] - 700 Sold_1[28] + 700 Lost_0[28] + 700 Lost_1[28]
   + 300 Left_1[28] + profits_vars[28] = 0
 Fraction_lower_bound_29: Fraction[29] >= 0
 Fraction_upper_bound_29: Fraction[29] <= 1
 R292: - 1.739788174204893 alphas[0] + 0.0187670479532857 alphas[1]
   - 0.5917651235030812 alphas[2] - 1.499247698737141 alphas[3] - alphas[4]
   + f_var[29] = 0
 R293: - 2433.451666584996 Fraction[29] + Q0_var[29] = 0
 R294: Q0_var[29] + Lost_0_expr_29 = 1829.854034014646
 R295: Q0_var[29] + Q_hat_adjusted_29 = 2315.995960431029
 R296: Sold_0[29] - Q0_var[29] - Q1_var[29] + total_stock_second_period_29
   = 0
 R297: total_stock_second_period_29 + Lost_1_expr_29 = 439.3795851933521
 R298: Sold_1[29] + Left_1[29] - total_stock_second_period_29 = 0
 R299: - 700 Sold_0[29] - 700 Sold_1[29] + 700 Lost_0[29] + 700 Lost_1[29]
   + 300 Left_1[29] + profits_vars[29] = 0
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
 Q0_var[0] <= 2434.451666584996
 Q0_var[1] <= 2434.451666584996
 Q0_var[2] <= 2434.451666584996
 Q0_var[3] <= 2434.451666584996
 Q0_var[4] <= 2434.451666584996
 Q0_var[5] <= 2434.451666584996
 Q0_var[6] <= 2434.451666584996
 Q0_var[7] <= 2434.451666584996
 Q0_var[8] <= 2434.451666584996
 Q0_var[9] <= 2434.451666584996
 Q0_var[10] <= 2434.451666584996
 Q0_var[11] <= 2434.451666584996
 Q0_var[12] <= 2434.451666584996
 Q0_var[13] <= 2434.451666584996
 Q0_var[14] <= 2434.451666584996
 Q0_var[15] <= 2434.451666584996
 Q0_var[16] <= 2434.451666584996
 Q0_var[17] <= 2434.451666584996
 Q0_var[18] <= 2434.451666584996
 Q0_var[19] <= 2434.451666584996
 Q0_var[20] <= 2434.451666584996
 Q0_var[21] <= 2434.451666584996
 Q0_var[22] <= 2434.451666584996
 Q0_var[23] <= 2434.451666584996
 Q0_var[24] <= 2434.451666584996
 Q0_var[25] <= 2434.451666584996
 Q0_var[26] <= 2434.451666584996
 Q0_var[27] <= 2434.451666584996
 Q0_var[28] <= 2434.451666584996
 Q0_var[29] <= 2434.451666584996
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
 GC0: Fraction[0] = LOGISTIC ( f_var[0] )
 min_sales_constr_0: Sold_0[0] = MIN ( Q0_var[0] , 2592.957502623658 )
 max_lost_constr_0: Lost_0[0] = MAX ( Lost_0_expr_0 , 0 )
 max_Q1_constr_0: Q1_var[0] = MAX ( Q_hat_adjusted_0 , 0 )
 min_sales2_constr_0: Sold_1[0] = MIN ( total_stock_second_period_0
   , 468.3804544210274 )
 max_lost2_constr_0: Lost_1[0] = MAX ( Lost_1_expr_0 , 0 )
 GC6: Fraction[1] = LOGISTIC ( f_var[1] )
 min_sales_constr_1: Sold_0[1] = MIN ( Q0_var[1] , 527.6091202858785 )
 max_lost_constr_1: Lost_0[1] = MAX ( Lost_0_expr_1 , 0 )
 max_Q1_constr_1: Q1_var[1] = MAX ( Q_hat_adjusted_1 , 0 )
 min_sales2_constr_1: Sold_1[1] = MIN ( total_stock_second_period_1
   , 268.6975352627283 )
 max_lost2_constr_1: Lost_1[1] = MAX ( Lost_1_expr_1 , 0 )
 GC12: Fraction[2] = LOGISTIC ( f_var[2] )
 min_sales_constr_2: Sold_0[2] = MIN ( Q0_var[2] , 1253.414429582754 )
 max_lost_constr_2: Lost_0[2] = MAX ( Lost_0_expr_2 , 0 )
 max_Q1_constr_2: Q1_var[2] = MAX ( Q_hat_adjusted_2 , 0 )
 min_sales2_constr_2: Sold_1[2] = MIN ( total_stock_second_period_2
   , 61.85760523940101 )
 max_lost2_constr_2: Lost_1[2] = MAX ( Lost_1_expr_2 , 0 )
 GC18: Fraction[3] = LOGISTIC ( f_var[3] )
 min_sales_constr_3: Sold_0[3] = MIN ( Q0_var[3] , 927.3474209617464 )
 max_lost_constr_3: Lost_0[3] = MAX ( Lost_0_expr_3 , 0 )
 max_Q1_constr_3: Q1_var[3] = MAX ( Q_hat_adjusted_3 , 0 )
 min_sales2_constr_3: Sold_1[3] = MIN ( total_stock_second_period_3
   , 762.717631850279 )
 max_lost2_constr_3: Lost_1[3] = MAX ( Lost_1_expr_3 , 0 )
 GC24: Fraction[4] = LOGISTIC ( f_var[4] )
 min_sales_constr_4: Sold_0[4] = MIN ( Q0_var[4] , 2258.646338916769 )
 max_lost_constr_4: Lost_0[4] = MAX ( Lost_0_expr_4 , 0 )
 max_Q1_constr_4: Q1_var[4] = MAX ( Q_hat_adjusted_4 , 0 )
 min_sales2_constr_4: Sold_1[4] = MIN ( total_stock_second_period_4
   , 543.5597790620841 )
 max_lost2_constr_4: Lost_1[4] = MAX ( Lost_1_expr_4 , 0 )
 GC30: Fraction[5] = LOGISTIC ( f_var[5] )
 min_sales_constr_5: Sold_0[5] = MIN ( Q0_var[5] , 2895.569709255679 )
 max_lost_constr_5: Lost_0[5] = MAX ( Lost_0_expr_5 , 0 )
 max_Q1_constr_5: Q1_var[5] = MAX ( Q_hat_adjusted_5 , 0 )
 min_sales2_constr_5: Sold_1[5] = MIN ( total_stock_second_period_5
   , 641.6202046038665 )
 max_lost2_constr_5: Lost_1[5] = MAX ( Lost_1_expr_5 , 0 )
 GC36: Fraction[6] = LOGISTIC ( f_var[6] )
 min_sales_constr_6: Sold_0[6] = MIN ( Q0_var[6] , 2267.079241103794 )
 max_lost_constr_6: Lost_0[6] = MAX ( Lost_0_expr_6 , 0 )
 max_Q1_constr_6: Q1_var[6] = MAX ( Q_hat_adjusted_6 , 0 )
 min_sales2_constr_6: Sold_1[6] = MIN ( total_stock_second_period_6
   , 665.980137395507 )
 max_lost2_constr_6: Lost_1[6] = MAX ( Lost_1_expr_6 , 0 )
 GC42: Fraction[7] = LOGISTIC ( f_var[7] )
 min_sales_constr_7: Sold_0[7] = MIN ( Q0_var[7] , 1693.557791064872 )
 max_lost_constr_7: Lost_0[7] = MAX ( Lost_0_expr_7 , 0 )
 max_Q1_constr_7: Q1_var[7] = MAX ( Q_hat_adjusted_7 , 0 )
 min_sales2_constr_7: Sold_1[7] = MIN ( total_stock_second_period_7
   , 25.63723783253555 )
 max_lost2_constr_7: Lost_1[7] = MAX ( Lost_1_expr_7 , 0 )
 GC48: Fraction[8] = LOGISTIC ( f_var[8] )
 min_sales_constr_8: Sold_0[8] = MIN ( Q0_var[8] , 1027.357280852033 )
 max_lost_constr_8: Lost_0[8] = MAX ( Lost_0_expr_8 , 0 )
 max_Q1_constr_8: Q1_var[8] = MAX ( Q_hat_adjusted_8 , 0 )
 min_sales2_constr_8: Sold_1[8] = MIN ( total_stock_second_period_8
   , 280.6622024003613 )
 max_lost2_constr_8: Lost_1[8] = MAX ( Lost_1_expr_8 , 0 )
 GC54: Fraction[9] = LOGISTIC ( f_var[9] )
 min_sales_constr_9: Sold_0[9] = MIN ( Q0_var[9] , 2172.665078693703 )
 max_lost_constr_9: Lost_0[9] = MAX ( Lost_0_expr_9 , 0 )
 max_Q1_constr_9: Q1_var[9] = MAX ( Q_hat_adjusted_9 , 0 )
 min_sales2_constr_9: Sold_1[9] = MIN ( total_stock_second_period_9
   , 371.1052747784937 )
 max_lost2_constr_9: Lost_1[9] = MAX ( Lost_1_expr_9 , 0 )
 GC60: Fraction[10] = LOGISTIC ( f_var[10] )
 min_sales_constr_10: Sold_0[10] = MIN ( Q0_var[10] , 1493.506928874833 )
 max_lost_constr_10: Lost_0[10] = MAX ( Lost_0_expr_10 , 0 )
 max_Q1_constr_10: Q1_var[10] = MAX ( Q_hat_adjusted_10 , 0 )
 min_sales2_constr_10: Sold_1[10] = MIN ( total_stock_second_period_10
   , 316.4556753408243 )
 max_lost2_constr_10: Lost_1[10] = MAX ( Lost_1_expr_10 , 0 )
 GC66: Fraction[11] = LOGISTIC ( f_var[11] )
 min_sales_constr_11: Sold_0[11] = MIN ( Q0_var[11] , 1592.663883481324 )
 max_lost_constr_11: Lost_0[11] = MAX ( Lost_0_expr_11 , 0 )
 max_Q1_constr_11: Q1_var[11] = MAX ( Q_hat_adjusted_11 , 0 )
 min_sales2_constr_11: Sold_1[11] = MIN ( total_stock_second_period_11
   , 313.9071128191609 )
 max_lost2_constr_11: Lost_1[11] = MAX ( Lost_1_expr_11 , 0 )
 GC72: Fraction[12] = LOGISTIC ( f_var[12] )
 min_sales_constr_12: Sold_0[12] = MIN ( Q0_var[12] , 2230.347415439984 )
 max_lost_constr_12: Lost_0[12] = MAX ( Lost_0_expr_12 , 0 )
 max_Q1_constr_12: Q1_var[12] = MAX ( Q_hat_adjusted_12 , 0 )
 min_sales2_constr_12: Sold_1[12] = MIN ( total_stock_second_period_12
   , 622.1450259213642 )
 max_lost2_constr_12: Lost_1[12] = MAX ( Lost_1_expr_12 , 0 )
 GC78: Fraction[13] = LOGISTIC ( f_var[13] )
 min_sales_constr_13: Sold_0[13] = MIN ( Q0_var[13] , 1481.786657512529 )
 max_lost_constr_13: Lost_0[13] = MAX ( Lost_0_expr_13 , 0 )
 max_Q1_constr_13: Q1_var[13] = MAX ( Q_hat_adjusted_13 , 0 )
 min_sales2_constr_13: Sold_1[13] = MIN ( total_stock_second_period_13
   , 399.05533900029 )
 max_lost2_constr_13: Lost_1[13] = MAX ( Lost_1_expr_13 , 0 )
 GC84: Fraction[14] = LOGISTIC ( f_var[14] )
 min_sales_constr_14: Sold_0[14] = MIN ( Q0_var[14] , 2020.616520896928 )
 max_lost_constr_14: Lost_0[14] = MAX ( Lost_0_expr_14 , 0 )
 max_Q1_constr_14: Q1_var[14] = MAX ( Q_hat_adjusted_14 , 0 )
 min_sales2_constr_14: Sold_1[14] = MIN ( total_stock_second_period_14
   , 293.1924253825994 )
 max_lost2_constr_14: Lost_1[14] = MAX ( Lost_1_expr_14 , 0 )
 GC90: Fraction[15] = LOGISTIC ( f_var[15] )
 min_sales_constr_15: Sold_0[15] = MIN ( Q0_var[15] , 1602.318516547827 )
 max_lost_constr_15: Lost_0[15] = MAX ( Lost_0_expr_15 , 0 )
 max_Q1_constr_15: Q1_var[15] = MAX ( Q_hat_adjusted_15 , 0 )
 min_sales2_constr_15: Sold_1[15] = MIN ( total_stock_second_period_15
   , 554.1257998625251 )
 max_lost2_constr_15: Lost_1[15] = MAX ( Lost_1_expr_15 , 0 )
 GC96: Fraction[16] = LOGISTIC ( f_var[16] )
 min_sales_constr_16: Sold_0[16] = MIN ( Q0_var[16] , 1633.236286342205 )
 max_lost_constr_16: Lost_0[16] = MAX ( Lost_0_expr_16 , 0 )
 max_Q1_constr_16: Q1_var[16] = MAX ( Q_hat_adjusted_16 , 0 )
 min_sales2_constr_16: Sold_1[16] = MIN ( total_stock_second_period_16
   , 810.1495778715318 )
 max_lost2_constr_16: Lost_1[16] = MAX ( Lost_1_expr_16 , 0 )
 GC102: Fraction[17] = LOGISTIC ( f_var[17] )
 min_sales_constr_17: Sold_0[17] = MIN ( Q0_var[17] , 1764.326703923937 )
 max_lost_constr_17: Lost_0[17] = MAX ( Lost_0_expr_17 , 0 )
 max_Q1_constr_17: Q1_var[17] = MAX ( Q_hat_adjusted_17 , 0 )
 min_sales2_constr_17: Sold_1[17] = MIN ( total_stock_second_period_17
   , 496.405442128 )
 max_lost2_constr_17: Lost_1[17] = MAX ( Lost_1_expr_17 , 0 )
 GC108: Fraction[18] = LOGISTIC ( f_var[18] )
 min_sales_constr_18: Sold_0[18] = MIN ( Q0_var[18] , 1255.703818018699 )
 max_lost_constr_18: Lost_0[18] = MAX ( Lost_0_expr_18 , 0 )
 max_Q1_constr_18: Q1_var[18] = MAX ( Q_hat_adjusted_18 , 0 )
 min_sales2_constr_18: Sold_1[18] = MIN ( total_stock_second_period_18
   , 270.7629070749643 )
 max_lost2_constr_18: Lost_1[18] = MAX ( Lost_1_expr_18 , 0 )
 GC114: Fraction[19] = LOGISTIC ( f_var[19] )
 min_sales_constr_19: Sold_0[19] = MIN ( Q0_var[19] , 1038.6059163213 )
 max_lost_constr_19: Lost_0[19] = MAX ( Lost_0_expr_19 , 0 )
 max_Q1_constr_19: Q1_var[19] = MAX ( Q_hat_adjusted_19 , 0 )
 min_sales2_constr_19: Sold_1[19] = MIN ( total_stock_second_period_19
   , 418.2221889975829 )
 max_lost2_constr_19: Lost_1[19] = MAX ( Lost_1_expr_19 , 0 )
 GC120: Fraction[20] = LOGISTIC ( f_var[20] )
 min_sales_constr_20: Sold_0[20] = MIN ( Q0_var[20] , 518.7184646007953 )
 max_lost_constr_20: Lost_0[20] = MAX ( Lost_0_expr_20 , 0 )
 max_Q1_constr_20: Q1_var[20] = MAX ( Q_hat_adjusted_20 , 0 )
 min_sales2_constr_20: Sold_1[20] = MIN ( total_stock_second_period_20
   , 566.9280970338278 )
 max_lost2_constr_20: Lost_1[20] = MAX ( Lost_1_expr_20 , 0 )
 GC126: Fraction[21] = LOGISTIC ( f_var[21] )
 min_sales_constr_21: Sold_0[21] = MIN ( Q0_var[21] , 1468.173150442183 )
 max_lost_constr_21: Lost_0[21] = MAX ( Lost_0_expr_21 , 0 )
 max_Q1_constr_21: Q1_var[21] = MAX ( Q_hat_adjusted_21 , 0 )
 min_sales2_constr_21: Sold_1[21] = MIN ( total_stock_second_period_21
   , 432.0364270309086 )
 max_lost2_constr_21: Lost_1[21] = MAX ( Lost_1_expr_21 , 0 )
 GC132: Fraction[22] = LOGISTIC ( f_var[22] )
 min_sales_constr_22: Sold_0[22] = MIN ( Q0_var[22] , 2362.440572219673 )
 max_lost_constr_22: Lost_0[22] = MAX ( Lost_0_expr_22 , 0 )
 max_Q1_constr_22: Q1_var[22] = MAX ( Q_hat_adjusted_22 , 0 )
 min_sales2_constr_22: Sold_1[22] = MIN ( total_stock_second_period_22
   , 260.9246159553333 )
 max_lost2_constr_22: Lost_1[22] = MAX ( Lost_1_expr_22 , 0 )
 GC138: Fraction[23] = LOGISTIC ( f_var[23] )
 min_sales_constr_23: Sold_0[23] = MIN ( Q0_var[23] , 2000.47567035108 )
 max_lost_constr_23: Lost_0[23] = MAX ( Lost_0_expr_23 , 0 )
 max_Q1_constr_23: Q1_var[23] = MAX ( Q_hat_adjusted_23 , 0 )
 min_sales2_constr_23: Sold_1[23] = MIN ( total_stock_second_period_23
   , 528.5532123430869 )
 max_lost2_constr_23: Lost_1[23] = MAX ( Lost_1_expr_23 , 0 )
 GC144: Fraction[24] = LOGISTIC ( f_var[24] )
 min_sales_constr_24: Sold_0[24] = MIN ( Q0_var[24] , 2455.346290371662 )
 max_lost_constr_24: Lost_0[24] = MAX ( Lost_0_expr_24 , 0 )
 max_Q1_constr_24: Q1_var[24] = MAX ( Q_hat_adjusted_24 , 0 )
 min_sales2_constr_24: Sold_1[24] = MIN ( total_stock_second_period_24
   , 301.2664997991592 )
 max_lost2_constr_24: Lost_1[24] = MAX ( Lost_1_expr_24 , 0 )
 GC150: Fraction[25] = LOGISTIC ( f_var[25] )
 min_sales_constr_25: Sold_0[25] = MIN ( Q0_var[25] , 2144.833376865706 )
 max_lost_constr_25: Lost_0[25] = MAX ( Lost_0_expr_25 , 0 )
 max_Q1_constr_25: Q1_var[25] = MAX ( Q_hat_adjusted_25 , 0 )
 min_sales2_constr_25: Sold_1[25] = MIN ( total_stock_second_period_25
   , 652.8604790557135 )
 max_lost2_constr_25: Lost_1[25] = MAX ( Lost_1_expr_25 , 0 )
 GC156: Fraction[26] = LOGISTIC ( f_var[26] )
 min_sales_constr_26: Sold_0[26] = MIN ( Q0_var[26] , 825.7637533755192 )
 max_lost_constr_26: Lost_0[26] = MAX ( Lost_0_expr_26 , 0 )
 max_Q1_constr_26: Q1_var[26] = MAX ( Q_hat_adjusted_26 , 0 )
 min_sales2_constr_26: Sold_1[26] = MIN ( total_stock_second_period_26
   , 322.0635548971552 )
 max_lost2_constr_26: Lost_1[26] = MAX ( Lost_1_expr_26 , 0 )
 GC162: Fraction[27] = LOGISTIC ( f_var[27] )
 min_sales_constr_27: Sold_0[27] = MIN ( Q0_var[27] , 1340.999558839349 )
 max_lost_constr_27: Lost_0[27] = MAX ( Lost_0_expr_27 , 0 )
 max_Q1_constr_27: Q1_var[27] = MAX ( Q_hat_adjusted_27 , 0 )
 min_sales2_constr_27: Sold_1[27] = MIN ( total_stock_second_period_27
   , 286.8666011770293 )
 max_lost2_constr_27: Lost_1[27] = MAX ( Lost_1_expr_27 , 0 )
 GC168: Fraction[28] = LOGISTIC ( f_var[28] )
 min_sales_constr_28: Sold_0[28] = MIN ( Q0_var[28] , 602.5924612900915 )
 max_lost_constr_28: Lost_0[28] = MAX ( Lost_0_expr_28 , 0 )
 max_Q1_constr_28: Q1_var[28] = MAX ( Q_hat_adjusted_28 , 0 )
 min_sales2_constr_28: Sold_1[28] = MIN ( total_stock_second_period_28
   , 318.6459754579744 )
 max_lost2_constr_28: Lost_1[28] = MAX ( Lost_1_expr_28 , 0 )
 GC174: Fraction[29] = LOGISTIC ( f_var[29] )
 min_sales_constr_29: Sold_0[29] = MIN ( Q0_var[29] , 1829.854034014646 )
 max_lost_constr_29: Lost_0[29] = MAX ( Lost_0_expr_29 , 0 )
 max_Q1_constr_29: Q1_var[29] = MAX ( Q_hat_adjusted_29 , 0 )
 min_sales2_constr_29: Sold_1[29] = MIN ( total_stock_second_period_29
   , 439.3795851933521 )
 max_lost2_constr_29: Lost_1[29] = MAX ( Lost_1_expr_29 , 0 )
End
