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
 R2: - 1.541253849845814 alphas[0] - 2.019990581899547 alphas[1]
   - 0.4082936743355209 alphas[2] - 0.1538371179486136 alphas[3]
   - alphas[4] + f_var[0] = 0
 R3: - 1829.241320246946 Fraction[0] + Q0_var[0] = 0
 R4: Q0_var[0] + Lost_0_expr_0 = 1829.685420803608
 R5: Q0_var[0] + Q_hat_adjusted_0 = 2288.185605884337
 R6: Sold_0[0] - Q0_var[0] - Q1_var[0] + total_stock_second_period_0 = 0
 R7: total_stock_second_period_0 + Lost_1_expr_0 = 465.5829409668786
 R8: Sold_1[0] + Left_1[0] - total_stock_second_period_0 = 0
 R9: - 500 Sold_0[0] - 500 Sold_1[0] + 500 Lost_0[0] + 500 Lost_1[0]
   + 500 Left_1[0] + profits_vars[0] = 0
 Fraction_lower_bound_1: Fraction[1] >= 0
 Fraction_upper_bound_1: Fraction[1] <= 1
 R12: 1.992898804628819 alphas[0] - 1.112594295343697 alphas[1]
   - 1.107356935299945 alphas[2] - 1.64244444220325 alphas[3] - alphas[4]
   + f_var[1] = 0
 R13: - 1829.241320246946 Fraction[1] + Q0_var[1] = 0
 R14: Q0_var[1] + Lost_0_expr_1 = 1516.012594089963
 R15: Q0_var[1] + Q_hat_adjusted_1 = 1912.187378370304
 R16: Sold_0[1] - Q0_var[1] - Q1_var[1] + total_stock_second_period_1 = 0
 R17: total_stock_second_period_1 + Lost_1_expr_1 = 399.2505165021991
 R18: Sold_1[1] + Left_1[1] - total_stock_second_period_1 = 0
 R19: - 500 Sold_0[1] - 500 Sold_1[1] + 500 Lost_0[1] + 500 Lost_1[1]
   + 500 Left_1[1] + profits_vars[1] = 0
 Fraction_lower_bound_2: Fraction[2] >= 0
 Fraction_upper_bound_2: Fraction[2] <= 1
 R22: - 1.483701890800565 alphas[0] - 0.2404841598113751 alphas[1]
   + 1.697978577377691 alphas[2] + 0.2465137999969989 alphas[3] - alphas[4]
   + f_var[2] = 0
 R23: - 1829.241320246946 Fraction[2] + Q0_var[2] = 0
 R24: Q0_var[2] + Lost_0_expr_2 = 1642.930953134821
 R25: Q0_var[2] + Q_hat_adjusted_2 = 2040.224290239275
 R26: Sold_0[2] - Q0_var[2] - Q1_var[2] + total_stock_second_period_2 = 0
 R27: total_stock_second_period_2 + Lost_1_expr_2 = 394.6852181822097
 R28: Sold_1[2] + Left_1[2] - total_stock_second_period_2 = 0
 R29: - 500 Sold_0[2] - 500 Sold_1[2] + 500 Lost_0[2] + 500 Lost_1[2]
   + 500 Left_1[2] + profits_vars[2] = 0
 Fraction_lower_bound_3: Fraction[3] >= 0
 Fraction_upper_bound_3: Fraction[3] <= 1
 R32: - 1.337216120762022 alphas[0] + 0.9040950082146765 alphas[1]
   + 1.386108656938341 alphas[2] - 1.368202138528298 alphas[3] - alphas[4]
   + f_var[3] = 0
 R33: - 1829.241320246946 Fraction[3] + Q0_var[3] = 0
 R34: Q0_var[3] + Lost_0_expr_3 = 1690.211236960094
 R35: Q0_var[3] + Q_hat_adjusted_3 = 2118.780207666051
 R36: Sold_0[3] - Q0_var[3] - Q1_var[3] + total_stock_second_period_3 = 0
 R37: total_stock_second_period_3 + Lost_1_expr_3 = 418.7431515184811
 R38: Sold_1[3] + Left_1[3] - total_stock_second_period_3 = 0
 R39: - 500 Sold_0[3] - 500 Sold_1[3] + 500 Lost_0[3] + 500 Lost_1[3]
   + 500 Left_1[3] + profits_vars[3] = 0
 Fraction_lower_bound_4: Fraction[4] >= 0
 Fraction_upper_bound_4: Fraction[4] <= 1
 R42: - 0.237881323713155 alphas[0] + 1.249411038760707 alphas[1]
   + 0.0902475462624092 alphas[2] + 0.8909197734323908 alphas[3]
   - alphas[4] + f_var[4] = 0
 R43: - 1829.241320246946 Fraction[4] + Q0_var[4] = 0
 R44: Q0_var[4] + Lost_0_expr_4 = 1257.849020537764
 R45: Q0_var[4] + Q_hat_adjusted_4 = 1581.615470746994
 R46: Sold_0[4] - Q0_var[4] - Q1_var[4] + total_stock_second_period_4 = 0
 R47: total_stock_second_period_4 + Lost_1_expr_4 = 327.8597214417377
 R48: Sold_1[4] + Left_1[4] - total_stock_second_period_4 = 0
 R49: - 500 Sold_0[4] - 500 Sold_1[4] + 500 Lost_0[4] + 500 Lost_1[4]
   + 500 Left_1[4] + profits_vars[4] = 0
 Fraction_lower_bound_5: Fraction[5] >= 0
 Fraction_upper_bound_5: Fraction[5] <= 1
 R52: - 0.2608669311813438 alphas[0] + 0.8234246579716354 alphas[1]
   + 0.2170664695893739 alphas[2] - 1.30359192641576 alphas[3] - alphas[4]
   + f_var[5] = 0
 R53: - 1829.241320246946 Fraction[5] + Q0_var[5] = 0
 R54: Q0_var[5] + Lost_0_expr_5 = 1559.669583497634
 R55: Q0_var[5] + Q_hat_adjusted_5 = 1946.604139485471
 R56: Sold_0[5] - Q0_var[5] - Q1_var[5] + total_stock_second_period_5 = 0
 R57: total_stock_second_period_5 + Lost_1_expr_5 = 381.0267677734467
 R58: Sold_1[5] + Left_1[5] - total_stock_second_period_5 = 0
 R59: - 500 Sold_0[5] - 500 Sold_1[5] + 500 Lost_0[5] + 500 Lost_1[5]
   + 500 Left_1[5] + profits_vars[5] = 0
 Fraction_lower_bound_6: Fraction[6] >= 0
 Fraction_upper_bound_6: Fraction[6] <= 1
 R62: - 0.9794235143501542 alphas[0] + 1.047396572258944 alphas[1]
   + 0.1956102706870147 alphas[2] + 0.4125050955682381 alphas[3]
   - alphas[4] + f_var[6] = 0
 R63: - 1829.241320246946 Fraction[6] + Q0_var[6] = 0
 R64: Q0_var[6] + Lost_0_expr_6 = 1440.37041360398
 R65: Q0_var[6] + Q_hat_adjusted_6 = 1800.006561218228
 R66: Sold_0[6] - Q0_var[6] - Q1_var[6] + total_stock_second_period_6 = 0
 R67: total_stock_second_period_6 + Lost_1_expr_6 = 360.9287237312172
 R68: Sold_1[6] + Left_1[6] - total_stock_second_period_6 = 0
 R69: - 500 Sold_0[6] - 500 Sold_1[6] + 500 Lost_0[6] + 500 Lost_1[6]
   + 500 Left_1[6] + profits_vars[6] = 0
 Fraction_lower_bound_7: Fraction[7] >= 0
 Fraction_upper_bound_7: Fraction[7] <= 1
 R72: 0.8412023743235226 alphas[0] + 0.2748705245454496 alphas[1]
   - 1.126748281959851 alphas[2] + 0.2780283782316132 alphas[3] - alphas[4]
   + f_var[7] = 0
 R73: - 1829.241320246946 Fraction[7] + Q0_var[7] = 0
 R74: Q0_var[7] + Lost_0_expr_7 = 1285.411795231588
 R75: Q0_var[7] + Q_hat_adjusted_7 = 1618.045786699398
 R76: Sold_0[7] - Q0_var[7] - Q1_var[7] + total_stock_second_period_7 = 0
 R77: total_stock_second_period_7 + Lost_1_expr_7 = 334.3886245152671
 R78: Sold_1[7] + Left_1[7] - total_stock_second_period_7 = 0
 R79: - 500 Sold_0[7] - 500 Sold_1[7] + 500 Lost_0[7] + 500 Lost_1[7]
   + 500 Left_1[7] + profits_vars[7] = 0
 Fraction_lower_bound_8: Fraction[8] >= 0
 Fraction_upper_bound_8: Fraction[8] <= 1
 R82: 0.8250289196442896 alphas[0] + 1.369727524695598 alphas[1]
   - 1.555828833523332 alphas[2] + 0.1507512078129342 alphas[3] - alphas[4]
   + f_var[8] = 0
 R83: - 1829.241320246946 Fraction[8] + Q0_var[8] = 0
 R84: Q0_var[8] + Lost_0_expr_8 = 1200.344547039831
 R85: Q0_var[8] + Q_hat_adjusted_8 = 1512.298975027852
 R86: Sold_0[8] - Q0_var[8] - Q1_var[8] + total_stock_second_period_8 = 0
 R87: total_stock_second_period_8 + Lost_1_expr_8 = 312.2091541086405
 R88: Sold_1[8] + Left_1[8] - total_stock_second_period_8 = 0
 R89: - 500 Sold_0[8] - 500 Sold_1[8] + 500 Lost_0[8] + 500 Lost_1[8]
   + 500 Left_1[8] + profits_vars[8] = 0
 Fraction_lower_bound_9: Fraction[9] >= 0
 Fraction_upper_bound_9: Fraction[9] <= 1
 R92: - 0.5275284126148034 alphas[0] - 0.1438284086530026 alphas[1]
   + 1.501154537810019 alphas[2] - 1.374333751994383 alphas[3] - alphas[4]
   + f_var[9] = 0
 R93: - 1829.241320246946 Fraction[9] + Q0_var[9] = 0
 R94: Q0_var[9] + Lost_0_expr_9 = 1710.162046923443
 R95: Q0_var[9] + Q_hat_adjusted_9 = 2154.869305243548
 R96: Sold_0[9] - Q0_var[9] - Q1_var[9] + total_stock_second_period_9 = 0
 R97: total_stock_second_period_9 + Lost_1_expr_9 = 443.9936045854041
 R98: Sold_1[9] + Left_1[9] - total_stock_second_period_9 = 0
 R99: - 500 Sold_0[9] - 500 Sold_1[9] + 500 Lost_0[9] + 500 Lost_1[9]
   + 500 Left_1[9] + profits_vars[9] = 0
 Fraction_lower_bound_10: Fraction[10] >= 0
 Fraction_upper_bound_10: Fraction[10] <= 1
 R102: 0.038713139135696 alphas[0] - 0.9033099391806745 alphas[1]
   - 0.3001850844399854 alphas[2] - 0.8606952602454936 alphas[3]
   - alphas[4] + f_var[10] = 0
 R103: - 1829.241320246946 Fraction[10] + Q0_var[10] = 0
 R104: Q0_var[10] + Lost_0_expr_10 = 1627.443863505028
 R105: Q0_var[10] + Q_hat_adjusted_10 = 2052.62885978969
 R106: Sold_0[10] - Q0_var[10] - Q1_var[10] + total_stock_second_period_10
   = 0
 R107: total_stock_second_period_10 + Lost_1_expr_10 = 421.0180255664817
 R108: Sold_1[10] + Left_1[10] - total_stock_second_period_10 = 0
 R109: - 500 Sold_0[10] - 500 Sold_1[10] + 500 Lost_0[10] + 500 Lost_1[10]
   + 500 Left_1[10] + profits_vars[10] = 0
 Fraction_lower_bound_11: Fraction[11] >= 0
 Fraction_upper_bound_11: Fraction[11] <= 1
 R112: - 0.1494228261696841 alphas[0] + 1.058421988503832 alphas[1]
   - 0.1060784815895609 alphas[2] - 0.3960217080076868 alphas[3]
   - alphas[4] + f_var[11] = 0
 R113: - 1829.241320246946 Fraction[11] + Q0_var[11] = 0
 R114: Q0_var[11] + Lost_0_expr_11 = 1416.814567649922
 R115: Q0_var[11] + Q_hat_adjusted_11 = 1780.298433265899
 R116: Sold_0[11] - Q0_var[11] - Q1_var[11] + total_stock_second_period_11
   = 0
 R117: total_stock_second_period_11 + Lost_1_expr_11 = 372.9665117462234
 R118: Sold_1[11] + Left_1[11] - total_stock_second_period_11 = 0
 R119: - 500 Sold_0[11] - 500 Sold_1[11] + 500 Lost_0[11] + 500 Lost_1[11]
   + 500 Left_1[11] + profits_vars[11] = 0
 Fraction_lower_bound_12: Fraction[12] >= 0
 Fraction_upper_bound_12: Fraction[12] <= 1
 R122: 1.769515248111474 alphas[0] - 1.192601218300086 alphas[1]
   + 0.8558708337472838 alphas[2] - 1.112615756120183 alphas[3] - alphas[4]
   + f_var[12] = 0
 R123: - 1829.241320246946 Fraction[12] + Q0_var[12] = 0
 R124: Q0_var[12] + Lost_0_expr_12 = 1533.239342198989
 R125: Q0_var[12] + Q_hat_adjusted_12 = 1923.454291159974
 R126: Sold_0[12] - Q0_var[12] - Q1_var[12] + total_stock_second_period_12
   = 0
 R127: total_stock_second_period_12 + Lost_1_expr_12 = 389.4123342093365
 R128: Sold_1[12] + Left_1[12] - total_stock_second_period_12 = 0
 R129: - 500 Sold_0[12] - 500 Sold_1[12] + 500 Lost_0[12] + 500 Lost_1[12]
   + 500 Left_1[12] + profits_vars[12] = 0
 Fraction_lower_bound_13: Fraction[13] >= 0
 Fraction_upper_bound_13: Fraction[13] <= 1
 R132: 0.5407381482136011 alphas[0] - 0.8920734897627978 alphas[1]
   + 0.1504414039447525 alphas[2] - 1.160513196598033 alphas[3] - alphas[4]
   + f_var[13] = 0
 R133: - 1829.241320246946 Fraction[13] + Q0_var[13] = 0
 R134: Q0_var[13] + Lost_0_expr_13 = 1608.32400716075
 R135: Q0_var[13] + Q_hat_adjusted_13 = 2010.66149003091
 R136: Sold_0[13] - Q0_var[13] - Q1_var[13] + total_stock_second_period_13
   = 0
 R137: total_stock_second_period_13 + Lost_1_expr_13 = 407.7964839567852
 R138: Sold_1[13] + Left_1[13] - total_stock_second_period_13 = 0
 R139: - 500 Sold_0[13] - 500 Sold_1[13] + 500 Lost_0[13] + 500 Lost_1[13]
   + 500 Left_1[13] + profits_vars[13] = 0
 Fraction_lower_bound_14: Fraction[14] >= 0
 Fraction_upper_bound_14: Fraction[14] <= 1
 R142: - 0.7789685860329789 alphas[0] - 0.7237226264797559 alphas[1]
   - 0.0674462009232731 alphas[2] + 1.613193352825219 alphas[3] - alphas[4]
   + f_var[14] = 0
 R143: - 1829.241320246946 Fraction[14] + Q0_var[14] = 0
 R144: Q0_var[14] + Lost_0_expr_14 = 1403.901481367409
 R145: Q0_var[14] + Q_hat_adjusted_14 = 1759.845560689608
 R146: Sold_0[14] - Q0_var[14] - Q1_var[14] + total_stock_second_period_14
   = 0
 R147: total_stock_second_period_14 + Lost_1_expr_14 = 350.5552388767384
 R148: Sold_1[14] + Left_1[14] - total_stock_second_period_14 = 0
 R149: - 500 Sold_0[14] - 500 Sold_1[14] + 500 Lost_0[14] + 500 Lost_1[14]
   + 500 Left_1[14] + profits_vars[14] = 0
 Fraction_lower_bound_15: Fraction[15] >= 0
 Fraction_upper_bound_15: Fraction[15] <= 1
 R152: - 0.1308042153486798 alphas[0] - 0.4390934242476194 alphas[1]
   + 0.224750820811735 alphas[2] - 0.2051628180813711 alphas[3] - alphas[4]
   + f_var[15] = 0
 R153: - 1829.241320246946 Fraction[15] + Q0_var[15] = 0
 R154: Q0_var[15] + Lost_0_expr_15 = 1552.668572630034
 R155: Q0_var[15] + Q_hat_adjusted_15 = 1940.322638030631
 R156: Sold_0[15] - Q0_var[15] - Q1_var[15] + total_stock_second_period_15
   = 0
 R157: total_stock_second_period_15 + Lost_1_expr_15 = 391.2735895827847
 R158: Sold_1[15] + Left_1[15] - total_stock_second_period_15 = 0
 R159: - 500 Sold_0[15] - 500 Sold_1[15] + 500 Lost_0[15] + 500 Lost_1[15]
   + 500 Left_1[15] + profits_vars[15] = 0
 Fraction_lower_bound_16: Fraction[16] >= 0
 Fraction_upper_bound_16: Fraction[16] <= 1
 R162: - 0.8506580589289626 alphas[0] - 0.2486480699198531 alphas[1]
   + 0.5309397405277253 alphas[2] + 1.48725018766534 alphas[3] - alphas[4]
   + f_var[16] = 0
 R163: - 1829.241320246946 Fraction[16] + Q0_var[16] = 0
 R164: Q0_var[16] + Lost_0_expr_16 = 1437.507398928891
 R165: Q0_var[16] + Q_hat_adjusted_16 = 1797.727727191046
 R166: Sold_0[16] - Q0_var[16] - Q1_var[16] + total_stock_second_period_16
   = 0
 R167: total_stock_second_period_16 + Lost_1_expr_16 = 357.3430905185121
 R168: Sold_1[16] + Left_1[16] - total_stock_second_period_16 = 0
 R169: - 500 Sold_0[16] - 500 Sold_1[16] + 500 Lost_0[16] + 500 Lost_1[16]
   + 500 Left_1[16] + profits_vars[16] = 0
 Fraction_lower_bound_17: Fraction[17] >= 0
 Fraction_upper_bound_17: Fraction[17] <= 1
 R172: - 1.123316297134206 alphas[0] + 1.050894787376198 alphas[1]
   - 0.2710442765598579 alphas[2] + 1.074808636788597 alphas[3] - alphas[4]
   + f_var[17] = 0
 R173: - 1829.241320246946 Fraction[17] + Q0_var[17] = 0
 R174: Q0_var[17] + Lost_0_expr_17 = 1355.24993955765
 R175: Q0_var[17] + Q_hat_adjusted_17 = 1682.437185448538
 R176: Sold_0[17] - Q0_var[17] - Q1_var[17] + total_stock_second_period_17
   = 0
 R177: total_stock_second_period_17 + Lost_1_expr_17 = 333.6488018209786
 R178: Sold_1[17] + Left_1[17] - total_stock_second_period_17 = 0
 R179: - 500 Sold_0[17] - 500 Sold_1[17] + 500 Lost_0[17] + 500 Lost_1[17]
   + 500 Left_1[17] + profits_vars[17] = 0
 Fraction_lower_bound_18: Fraction[18] >= 0
 Fraction_upper_bound_18: Fraction[18] <= 1
 R182: 0.7491012015205633 alphas[0] + 0.5334982307492885 alphas[1]
   - 1.362479139188685 alphas[2] + 1.435105839254642 alphas[3] - alphas[4]
   + f_var[18] = 0
 R183: - 1829.241320246946 Fraction[18] + Q0_var[18] = 0
 R184: Q0_var[18] + Lost_0_expr_18 = 1172.052768299522
 R185: Q0_var[18] + Q_hat_adjusted_18 = 1452.957921204186
 R186: Sold_0[18] - Q0_var[18] - Q1_var[18] + total_stock_second_period_18
   = 0
 R187: total_stock_second_period_18 + Lost_1_expr_18 = 277.1425329172206
 R188: Sold_1[18] + Left_1[18] - total_stock_second_period_18 = 0
 R189: - 500 Sold_0[18] - 500 Sold_1[18] + 500 Lost_0[18] + 500 Lost_1[18]
   + 500 Left_1[18] + profits_vars[18] = 0
 Fraction_lower_bound_19: Fraction[19] >= 0
 Fraction_upper_bound_19: Fraction[19] <= 1
 R192: - 0.3991178146662819 alphas[0] - 1.855475140457852 alphas[1]
   + 0.8241475768256888 alphas[2] + 0.5814073737540502 alphas[3]
   - alphas[4] + f_var[19] = 0
 R193: - 1829.241320246946 Fraction[19] + Q0_var[19] = 0
 R194: Q0_var[19] + Lost_0_expr_19 = 1629.556004938272
 R195: Q0_var[19] + Q_hat_adjusted_19 = 2042.012397854521
 R196: Sold_0[19] - Q0_var[19] - Q1_var[19] + total_stock_second_period_19
   = 0
 R197: total_stock_second_period_19 + Lost_1_expr_19 = 408.5160940880857
 R198: Sold_1[19] + Left_1[19] - total_stock_second_period_19 = 0
 R199: - 500 Sold_0[19] - 500 Sold_1[19] + 500 Lost_0[19] + 500 Lost_1[19]
   + 500 Left_1[19] + profits_vars[19] = 0
 Fraction_lower_bound_20: Fraction[20] >= 0
 Fraction_upper_bound_20: Fraction[20] <= 1
 R202: 1.415065403221863 alphas[0] - 0.6597746969314081 alphas[1]
   + 2.213270564167405 alphas[2] + 0.8786478363208375 alphas[3] - alphas[4]
   + f_var[20] = 0
 R203: - 1829.241320246946 Fraction[20] + Q0_var[20] = 0
 R204: Q0_var[20] + Lost_0_expr_20 = 1287.728823133599
 R205: Q0_var[20] + Q_hat_adjusted_20 = 1627.457681554218
 R206: Sold_0[20] - Q0_var[20] - Q1_var[20] + total_stock_second_period_20
   = 0
 R207: total_stock_second_period_20 + Lost_1_expr_20 = 333.3080956722663
 R208: Sold_1[20] + Left_1[20] - total_stock_second_period_20 = 0
 R209: - 500 Sold_0[20] - 500 Sold_1[20] + 500 Lost_0[20] + 500 Lost_1[20]
   + 500 Left_1[20] + profits_vars[20] = 0
 Fraction_lower_bound_21: Fraction[21] >= 0
 Fraction_upper_bound_21: Fraction[21] <= 1
 R212: 0.1278425830839482 alphas[0] + 1.030152477869554 alphas[1]
   - 1.40207155387602 alphas[2] - 0.486215275191437 alphas[3] - alphas[4]
   + f_var[21] = 0
 R213: - 1829.241320246946 Fraction[21] + Q0_var[21] = 0
 R214: Q0_var[21] + Lost_0_expr_21 = 1379.178835508619
 R215: Q0_var[21] + Q_hat_adjusted_21 = 1717.668503598654
 R216: Sold_0[21] - Q0_var[21] - Q1_var[21] + total_stock_second_period_21
   = 0
 R217: total_stock_second_period_21 + Lost_1_expr_21 = 336.9481704843326
 R218: Sold_1[21] + Left_1[21] - total_stock_second_period_21 = 0
 R219: - 500 Sold_0[21] - 500 Sold_1[21] + 500 Lost_0[21] + 500 Lost_1[21]
   + 500 Left_1[21] + profits_vars[21] = 0
 Fraction_lower_bound_22: Fraction[22] >= 0
 Fraction_upper_bound_22: Fraction[22] <= 1
 R222: 0.1127179511384918 alphas[0] - 0.6028604682640695 alphas[1]
   - 0.209324649687527 alphas[2] + 1.152806658653639 alphas[3] - alphas[4]
   + f_var[22] = 0
 R223: - 1829.241320246946 Fraction[22] + Q0_var[22] = 0
 R224: Q0_var[22] + Lost_0_expr_22 = 1389.961608116605
 R225: Q0_var[22] + Q_hat_adjusted_22 = 1741.409941754659
 R226: Sold_0[22] - Q0_var[22] - Q1_var[22] + total_stock_second_period_22
   = 0
 R227: total_stock_second_period_22 + Lost_1_expr_22 = 350.259320652215
 R228: Sold_1[22] + Left_1[22] - total_stock_second_period_22 = 0
 R229: - 500 Sold_0[22] - 500 Sold_1[22] + 500 Lost_0[22] + 500 Lost_1[22]
   + 500 Left_1[22] + profits_vars[22] = 0
 Fraction_lower_bound_23: Fraction[23] >= 0
 Fraction_upper_bound_23: Fraction[23] <= 1
 R232: 1.72852479386002 alphas[0] + 1.315966968302297 alphas[1]
   - 1.222949256516093 alphas[2] + 0.7839101778757477 alphas[3] - alphas[4]
   + f_var[23] = 0
 R233: - 1829.241320246946 Fraction[23] + Q0_var[23] = 0
 R234: Q0_var[23] + Lost_0_expr_23 = 1033.510955045644
 R235: Q0_var[23] + Q_hat_adjusted_23 = 1303.544560568157
 R236: Sold_0[23] - Q0_var[23] - Q1_var[23] + total_stock_second_period_23
   = 0
 R237: total_stock_second_period_23 + Lost_1_expr_23 = 269.8787122635263
 R238: Sold_1[23] + Left_1[23] - total_stock_second_period_23 = 0
 R239: - 500 Sold_0[23] - 500 Sold_1[23] + 500 Lost_0[23] + 500 Lost_1[23]
   + 500 Left_1[23] + profits_vars[23] = 0
 Fraction_lower_bound_24: Fraction[24] >= 0
 Fraction_upper_bound_24: Fraction[24] <= 1
 R242: - 0.3411887253336691 alphas[0] + 0.3765967400035515 alphas[1]
   - 0.7477806307898048 alphas[2] - 0.9222149268457147 alphas[3]
   - alphas[4] + f_var[24] = 0
 R243: - 1829.241320246946 Fraction[24] + Q0_var[24] = 0
 R244: Q0_var[24] + Lost_0_expr_24 = 1530.866129762609
 R245: Q0_var[24] + Q_hat_adjusted_24 = 1925.788093451509
 R246: Sold_0[24] - Q0_var[24] - Q1_var[24] + total_stock_second_period_24
   = 0
 R247: total_stock_second_period_24 + Lost_1_expr_24 = 401.6456708664231
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
 Q0_var[0] <= 1830.241320246946
 Q0_var[1] <= 1830.241320246946
 Q0_var[2] <= 1830.241320246946
 Q0_var[3] <= 1830.241320246946
 Q0_var[4] <= 1830.241320246946
 Q0_var[5] <= 1830.241320246946
 Q0_var[6] <= 1830.241320246946
 Q0_var[7] <= 1830.241320246946
 Q0_var[8] <= 1830.241320246946
 Q0_var[9] <= 1830.241320246946
 Q0_var[10] <= 1830.241320246946
 Q0_var[11] <= 1830.241320246946
 Q0_var[12] <= 1830.241320246946
 Q0_var[13] <= 1830.241320246946
 Q0_var[14] <= 1830.241320246946
 Q0_var[15] <= 1830.241320246946
 Q0_var[16] <= 1830.241320246946
 Q0_var[17] <= 1830.241320246946
 Q0_var[18] <= 1830.241320246946
 Q0_var[19] <= 1830.241320246946
 Q0_var[20] <= 1830.241320246946
 Q0_var[21] <= 1830.241320246946
 Q0_var[22] <= 1830.241320246946
 Q0_var[23] <= 1830.241320246946
 Q0_var[24] <= 1830.241320246946
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
 min_sales_constr_0: Sold_0[0] = MIN ( Q0_var[0] , 1829.685420803608 )
 max_lost_constr_0: Lost_0[0] = MAX ( Lost_0_expr_0 , 0 )
 max_Q1_constr_0: Q1_var[0] = MAX ( Q_hat_adjusted_0 , 0 )
 min_sales2_constr_0: Sold_1[0] = MIN ( total_stock_second_period_0
   , 465.5829409668786 )
 max_lost2_constr_0: Lost_1[0] = MAX ( Lost_1_expr_0 , 0 )
 GC6: Fraction[1] = LOGISTIC ( f_var[1] )
 min_sales_constr_1: Sold_0[1] = MIN ( Q0_var[1] , 1516.012594089963 )
 max_lost_constr_1: Lost_0[1] = MAX ( Lost_0_expr_1 , 0 )
 max_Q1_constr_1: Q1_var[1] = MAX ( Q_hat_adjusted_1 , 0 )
 min_sales2_constr_1: Sold_1[1] = MIN ( total_stock_second_period_1
   , 399.2505165021991 )
 max_lost2_constr_1: Lost_1[1] = MAX ( Lost_1_expr_1 , 0 )
 GC12: Fraction[2] = LOGISTIC ( f_var[2] )
 min_sales_constr_2: Sold_0[2] = MIN ( Q0_var[2] , 1642.930953134821 )
 max_lost_constr_2: Lost_0[2] = MAX ( Lost_0_expr_2 , 0 )
 max_Q1_constr_2: Q1_var[2] = MAX ( Q_hat_adjusted_2 , 0 )
 min_sales2_constr_2: Sold_1[2] = MIN ( total_stock_second_period_2
   , 394.6852181822097 )
 max_lost2_constr_2: Lost_1[2] = MAX ( Lost_1_expr_2 , 0 )
 GC18: Fraction[3] = LOGISTIC ( f_var[3] )
 min_sales_constr_3: Sold_0[3] = MIN ( Q0_var[3] , 1690.211236960094 )
 max_lost_constr_3: Lost_0[3] = MAX ( Lost_0_expr_3 , 0 )
 max_Q1_constr_3: Q1_var[3] = MAX ( Q_hat_adjusted_3 , 0 )
 min_sales2_constr_3: Sold_1[3] = MIN ( total_stock_second_period_3
   , 418.7431515184811 )
 max_lost2_constr_3: Lost_1[3] = MAX ( Lost_1_expr_3 , 0 )
 GC24: Fraction[4] = LOGISTIC ( f_var[4] )
 min_sales_constr_4: Sold_0[4] = MIN ( Q0_var[4] , 1257.849020537764 )
 max_lost_constr_4: Lost_0[4] = MAX ( Lost_0_expr_4 , 0 )
 max_Q1_constr_4: Q1_var[4] = MAX ( Q_hat_adjusted_4 , 0 )
 min_sales2_constr_4: Sold_1[4] = MIN ( total_stock_second_period_4
   , 327.8597214417377 )
 max_lost2_constr_4: Lost_1[4] = MAX ( Lost_1_expr_4 , 0 )
 GC30: Fraction[5] = LOGISTIC ( f_var[5] )
 min_sales_constr_5: Sold_0[5] = MIN ( Q0_var[5] , 1559.669583497634 )
 max_lost_constr_5: Lost_0[5] = MAX ( Lost_0_expr_5 , 0 )
 max_Q1_constr_5: Q1_var[5] = MAX ( Q_hat_adjusted_5 , 0 )
 min_sales2_constr_5: Sold_1[5] = MIN ( total_stock_second_period_5
   , 381.0267677734467 )
 max_lost2_constr_5: Lost_1[5] = MAX ( Lost_1_expr_5 , 0 )
 GC36: Fraction[6] = LOGISTIC ( f_var[6] )
 min_sales_constr_6: Sold_0[6] = MIN ( Q0_var[6] , 1440.37041360398 )
 max_lost_constr_6: Lost_0[6] = MAX ( Lost_0_expr_6 , 0 )
 max_Q1_constr_6: Q1_var[6] = MAX ( Q_hat_adjusted_6 , 0 )
 min_sales2_constr_6: Sold_1[6] = MIN ( total_stock_second_period_6
   , 360.9287237312172 )
 max_lost2_constr_6: Lost_1[6] = MAX ( Lost_1_expr_6 , 0 )
 GC42: Fraction[7] = LOGISTIC ( f_var[7] )
 min_sales_constr_7: Sold_0[7] = MIN ( Q0_var[7] , 1285.411795231588 )
 max_lost_constr_7: Lost_0[7] = MAX ( Lost_0_expr_7 , 0 )
 max_Q1_constr_7: Q1_var[7] = MAX ( Q_hat_adjusted_7 , 0 )
 min_sales2_constr_7: Sold_1[7] = MIN ( total_stock_second_period_7
   , 334.3886245152671 )
 max_lost2_constr_7: Lost_1[7] = MAX ( Lost_1_expr_7 , 0 )
 GC48: Fraction[8] = LOGISTIC ( f_var[8] )
 min_sales_constr_8: Sold_0[8] = MIN ( Q0_var[8] , 1200.344547039831 )
 max_lost_constr_8: Lost_0[8] = MAX ( Lost_0_expr_8 , 0 )
 max_Q1_constr_8: Q1_var[8] = MAX ( Q_hat_adjusted_8 , 0 )
 min_sales2_constr_8: Sold_1[8] = MIN ( total_stock_second_period_8
   , 312.2091541086405 )
 max_lost2_constr_8: Lost_1[8] = MAX ( Lost_1_expr_8 , 0 )
 GC54: Fraction[9] = LOGISTIC ( f_var[9] )
 min_sales_constr_9: Sold_0[9] = MIN ( Q0_var[9] , 1710.162046923443 )
 max_lost_constr_9: Lost_0[9] = MAX ( Lost_0_expr_9 , 0 )
 max_Q1_constr_9: Q1_var[9] = MAX ( Q_hat_adjusted_9 , 0 )
 min_sales2_constr_9: Sold_1[9] = MIN ( total_stock_second_period_9
   , 443.9936045854041 )
 max_lost2_constr_9: Lost_1[9] = MAX ( Lost_1_expr_9 , 0 )
 GC60: Fraction[10] = LOGISTIC ( f_var[10] )
 min_sales_constr_10: Sold_0[10] = MIN ( Q0_var[10] , 1627.443863505028 )
 max_lost_constr_10: Lost_0[10] = MAX ( Lost_0_expr_10 , 0 )
 max_Q1_constr_10: Q1_var[10] = MAX ( Q_hat_adjusted_10 , 0 )
 min_sales2_constr_10: Sold_1[10] = MIN ( total_stock_second_period_10
   , 421.0180255664817 )
 max_lost2_constr_10: Lost_1[10] = MAX ( Lost_1_expr_10 , 0 )
 GC66: Fraction[11] = LOGISTIC ( f_var[11] )
 min_sales_constr_11: Sold_0[11] = MIN ( Q0_var[11] , 1416.814567649922 )
 max_lost_constr_11: Lost_0[11] = MAX ( Lost_0_expr_11 , 0 )
 max_Q1_constr_11: Q1_var[11] = MAX ( Q_hat_adjusted_11 , 0 )
 min_sales2_constr_11: Sold_1[11] = MIN ( total_stock_second_period_11
   , 372.9665117462234 )
 max_lost2_constr_11: Lost_1[11] = MAX ( Lost_1_expr_11 , 0 )
 GC72: Fraction[12] = LOGISTIC ( f_var[12] )
 min_sales_constr_12: Sold_0[12] = MIN ( Q0_var[12] , 1533.239342198989 )
 max_lost_constr_12: Lost_0[12] = MAX ( Lost_0_expr_12 , 0 )
 max_Q1_constr_12: Q1_var[12] = MAX ( Q_hat_adjusted_12 , 0 )
 min_sales2_constr_12: Sold_1[12] = MIN ( total_stock_second_period_12
   , 389.4123342093365 )
 max_lost2_constr_12: Lost_1[12] = MAX ( Lost_1_expr_12 , 0 )
 GC78: Fraction[13] = LOGISTIC ( f_var[13] )
 min_sales_constr_13: Sold_0[13] = MIN ( Q0_var[13] , 1608.32400716075 )
 max_lost_constr_13: Lost_0[13] = MAX ( Lost_0_expr_13 , 0 )
 max_Q1_constr_13: Q1_var[13] = MAX ( Q_hat_adjusted_13 , 0 )
 min_sales2_constr_13: Sold_1[13] = MIN ( total_stock_second_period_13
   , 407.7964839567852 )
 max_lost2_constr_13: Lost_1[13] = MAX ( Lost_1_expr_13 , 0 )
 GC84: Fraction[14] = LOGISTIC ( f_var[14] )
 min_sales_constr_14: Sold_0[14] = MIN ( Q0_var[14] , 1403.901481367409 )
 max_lost_constr_14: Lost_0[14] = MAX ( Lost_0_expr_14 , 0 )
 max_Q1_constr_14: Q1_var[14] = MAX ( Q_hat_adjusted_14 , 0 )
 min_sales2_constr_14: Sold_1[14] = MIN ( total_stock_second_period_14
   , 350.5552388767384 )
 max_lost2_constr_14: Lost_1[14] = MAX ( Lost_1_expr_14 , 0 )
 GC90: Fraction[15] = LOGISTIC ( f_var[15] )
 min_sales_constr_15: Sold_0[15] = MIN ( Q0_var[15] , 1552.668572630034 )
 max_lost_constr_15: Lost_0[15] = MAX ( Lost_0_expr_15 , 0 )
 max_Q1_constr_15: Q1_var[15] = MAX ( Q_hat_adjusted_15 , 0 )
 min_sales2_constr_15: Sold_1[15] = MIN ( total_stock_second_period_15
   , 391.2735895827847 )
 max_lost2_constr_15: Lost_1[15] = MAX ( Lost_1_expr_15 , 0 )
 GC96: Fraction[16] = LOGISTIC ( f_var[16] )
 min_sales_constr_16: Sold_0[16] = MIN ( Q0_var[16] , 1437.507398928891 )
 max_lost_constr_16: Lost_0[16] = MAX ( Lost_0_expr_16 , 0 )
 max_Q1_constr_16: Q1_var[16] = MAX ( Q_hat_adjusted_16 , 0 )
 min_sales2_constr_16: Sold_1[16] = MIN ( total_stock_second_period_16
   , 357.3430905185121 )
 max_lost2_constr_16: Lost_1[16] = MAX ( Lost_1_expr_16 , 0 )
 GC102: Fraction[17] = LOGISTIC ( f_var[17] )
 min_sales_constr_17: Sold_0[17] = MIN ( Q0_var[17] , 1355.24993955765 )
 max_lost_constr_17: Lost_0[17] = MAX ( Lost_0_expr_17 , 0 )
 max_Q1_constr_17: Q1_var[17] = MAX ( Q_hat_adjusted_17 , 0 )
 min_sales2_constr_17: Sold_1[17] = MIN ( total_stock_second_period_17
   , 333.6488018209786 )
 max_lost2_constr_17: Lost_1[17] = MAX ( Lost_1_expr_17 , 0 )
 GC108: Fraction[18] = LOGISTIC ( f_var[18] )
 min_sales_constr_18: Sold_0[18] = MIN ( Q0_var[18] , 1172.052768299522 )
 max_lost_constr_18: Lost_0[18] = MAX ( Lost_0_expr_18 , 0 )
 max_Q1_constr_18: Q1_var[18] = MAX ( Q_hat_adjusted_18 , 0 )
 min_sales2_constr_18: Sold_1[18] = MIN ( total_stock_second_period_18
   , 277.1425329172206 )
 max_lost2_constr_18: Lost_1[18] = MAX ( Lost_1_expr_18 , 0 )
 GC114: Fraction[19] = LOGISTIC ( f_var[19] )
 min_sales_constr_19: Sold_0[19] = MIN ( Q0_var[19] , 1629.556004938272 )
 max_lost_constr_19: Lost_0[19] = MAX ( Lost_0_expr_19 , 0 )
 max_Q1_constr_19: Q1_var[19] = MAX ( Q_hat_adjusted_19 , 0 )
 min_sales2_constr_19: Sold_1[19] = MIN ( total_stock_second_period_19
   , 408.5160940880857 )
 max_lost2_constr_19: Lost_1[19] = MAX ( Lost_1_expr_19 , 0 )
 GC120: Fraction[20] = LOGISTIC ( f_var[20] )
 min_sales_constr_20: Sold_0[20] = MIN ( Q0_var[20] , 1287.728823133599 )
 max_lost_constr_20: Lost_0[20] = MAX ( Lost_0_expr_20 , 0 )
 max_Q1_constr_20: Q1_var[20] = MAX ( Q_hat_adjusted_20 , 0 )
 min_sales2_constr_20: Sold_1[20] = MIN ( total_stock_second_period_20
   , 333.3080956722663 )
 max_lost2_constr_20: Lost_1[20] = MAX ( Lost_1_expr_20 , 0 )
 GC126: Fraction[21] = LOGISTIC ( f_var[21] )
 min_sales_constr_21: Sold_0[21] = MIN ( Q0_var[21] , 1379.178835508619 )
 max_lost_constr_21: Lost_0[21] = MAX ( Lost_0_expr_21 , 0 )
 max_Q1_constr_21: Q1_var[21] = MAX ( Q_hat_adjusted_21 , 0 )
 min_sales2_constr_21: Sold_1[21] = MIN ( total_stock_second_period_21
   , 336.9481704843326 )
 max_lost2_constr_21: Lost_1[21] = MAX ( Lost_1_expr_21 , 0 )
 GC132: Fraction[22] = LOGISTIC ( f_var[22] )
 min_sales_constr_22: Sold_0[22] = MIN ( Q0_var[22] , 1389.961608116605 )
 max_lost_constr_22: Lost_0[22] = MAX ( Lost_0_expr_22 , 0 )
 max_Q1_constr_22: Q1_var[22] = MAX ( Q_hat_adjusted_22 , 0 )
 min_sales2_constr_22: Sold_1[22] = MIN ( total_stock_second_period_22
   , 350.259320652215 )
 max_lost2_constr_22: Lost_1[22] = MAX ( Lost_1_expr_22 , 0 )
 GC138: Fraction[23] = LOGISTIC ( f_var[23] )
 min_sales_constr_23: Sold_0[23] = MIN ( Q0_var[23] , 1033.510955045644 )
 max_lost_constr_23: Lost_0[23] = MAX ( Lost_0_expr_23 , 0 )
 max_Q1_constr_23: Q1_var[23] = MAX ( Q_hat_adjusted_23 , 0 )
 min_sales2_constr_23: Sold_1[23] = MIN ( total_stock_second_period_23
   , 269.8787122635263 )
 max_lost2_constr_23: Lost_1[23] = MAX ( Lost_1_expr_23 , 0 )
 GC144: Fraction[24] = LOGISTIC ( f_var[24] )
 min_sales_constr_24: Sold_0[24] = MIN ( Q0_var[24] , 1530.866129762609 )
 max_lost_constr_24: Lost_0[24] = MAX ( Lost_0_expr_24 , 0 )
 max_Q1_constr_24: Q1_var[24] = MAX ( Q_hat_adjusted_24 , 0 )
 min_sales2_constr_24: Sold_1[24] = MIN ( total_stock_second_period_24
   , 401.6456708664231 )
 max_lost2_constr_24: Lost_1[24] = MAX ( Lost_1_expr_24 , 0 )
End
