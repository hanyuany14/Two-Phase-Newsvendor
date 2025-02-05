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
 R2: 1.275151299821754 alphas[0] + 1.741126690859786 alphas[1]
   - 0.4601944167520682 alphas[2] - 0.6631229642549983 alphas[3]
   - alphas[4] + f_var[0] = 0
 R3: - 1313.714597627199 Fraction[0] + Q0_var[0] = 0
 R4: Q0_var[0] + Lost_0_expr_0 = 998.441375369316
 R5: Q0_var[0] + Q_hat_adjusted_0 = 1237.979975632998
 R6: Sold_0[0] - Q0_var[0] - Q1_var[0] + total_stock_second_period_0 = 0
 R7: total_stock_second_period_0 + Lost_1_expr_0 = 237.4976552775631
 R8: Sold_1[0] + Left_1[0] - total_stock_second_period_0 = 0
 R9: - 800 Sold_0[0] - 800 Sold_1[0] + 800 Lost_0[0] + 800 Lost_1[0]
   + 200 Left_1[0] + profits_vars[0] = 0
 Fraction_lower_bound_1: Fraction[1] >= 0
 Fraction_upper_bound_1: Fraction[1] <= 1
 R12: - 0.0629594742181593 alphas[0] - 0.0755625617877274 alphas[1]
   - 1.508604797874747 alphas[2] - 1.13534667895843 alphas[3] - alphas[4]
   + f_var[1] = 0
 R13: - 1313.714597627199 Fraction[1] + Q0_var[1] = 0
 R14: Q0_var[1] + Lost_0_expr_1 = 1185.759218896487
 R15: Q0_var[1] + Q_hat_adjusted_1 = 1484.236610340939
 R16: Sold_0[1] - Q0_var[1] - Q1_var[1] + total_stock_second_period_1 = 0
 R17: total_stock_second_period_1 + Lost_1_expr_1 = 296.3658779636506
 R18: Sold_1[1] + Left_1[1] - total_stock_second_period_1 = 0
 R19: - 800 Sold_0[1] - 800 Sold_1[1] + 800 Lost_0[1] + 800 Lost_1[1]
   + 200 Left_1[1] + profits_vars[1] = 0
 Fraction_lower_bound_2: Fraction[2] >= 0
 Fraction_upper_bound_2: Fraction[2] <= 1
 R22: - 0.3430378521939894 alphas[0] - 0.7775627931207666 alphas[1]
   + 0.6430523987474072 alphas[2] + 0.5871757688477869 alphas[3]
   - alphas[4] + f_var[2] = 0
 R23: - 1313.714597627199 Fraction[2] + Q0_var[2] = 0
 R24: Q0_var[2] + Lost_0_expr_2 = 610.2814917716669
 R25: Q0_var[2] + Q_hat_adjusted_2 = 788.0593741156265
 R26: Sold_0[2] - Q0_var[2] - Q1_var[2] + total_stock_second_period_2 = 0
 R27: total_stock_second_period_2 + Lost_1_expr_2 = 175.5503678586944
 R28: Sold_1[2] + Left_1[2] - total_stock_second_period_2 = 0
 R29: - 800 Sold_0[2] - 800 Sold_1[2] + 800 Lost_0[2] + 800 Lost_1[2]
   + 200 Left_1[2] + profits_vars[2] = 0
 Fraction_lower_bound_3: Fraction[3] >= 0
 Fraction_upper_bound_3: Fraction[3] <= 1
 R32: 1.21190371757587 alphas[0] - 1.174411438914283 alphas[1]
   + 1.691421071645701 alphas[2] - 0.6563781730980143 alphas[3] - alphas[4]
   + f_var[3] = 0
 R33: - 1313.714597627199 Fraction[3] + Q0_var[3] = 0
 R34: Q0_var[3] + Lost_0_expr_3 = 414.605069752064
 R35: Q0_var[3] + Q_hat_adjusted_3 = 533.4620174092303
 R36: Sold_0[3] - Q0_var[3] - Q1_var[3] + total_stock_second_period_3 = 0
 R37: total_stock_second_period_3 + Lost_1_expr_3 = 116.6844884669272
 R38: Sold_1[3] + Left_1[3] - total_stock_second_period_3 = 0
 R39: - 800 Sold_0[3] - 800 Sold_1[3] + 800 Lost_0[3] + 800 Lost_1[3]
   + 200 Left_1[3] + profits_vars[3] = 0
 Fraction_lower_bound_4: Fraction[4] >= 0
 Fraction_upper_bound_4: Fraction[4] <= 1
 R42: 1.642055463298931 alphas[0] + 1.020938901788023 alphas[1]
   - 1.392805764692453 alphas[2] + 1.203534446733136 alphas[3] - alphas[4]
   + f_var[4] = 0
 R43: - 1313.714597627199 Fraction[4] + Q0_var[4] = 0
 R44: Q0_var[4] + Lost_0_expr_4 = 1184.825383190634
 R45: Q0_var[4] + Q_hat_adjusted_4 = 1476.701979356353
 R46: Sold_0[4] - Q0_var[4] - Q1_var[4] + total_stock_second_period_4 = 0
 R47: total_stock_second_period_4 + Lost_1_expr_4 = 289.8637644831363
 R48: Sold_1[4] + Left_1[4] - total_stock_second_period_4 = 0
 R49: - 800 Sold_0[4] - 800 Sold_1[4] + 800 Lost_0[4] + 800 Lost_1[4]
   + 200 Left_1[4] + profits_vars[4] = 0
 Fraction_lower_bound_5: Fraction[5] >= 0
 Fraction_upper_bound_5: Fraction[5] <= 1
 R52: - 0.7430459476667776 alphas[0] - 1.513393448458082 alphas[1]
   + 1.312345335208576 alphas[2] - 1.255023324603177 alphas[3] - alphas[4]
   + f_var[5] = 0
 R53: - 1313.714597627199 Fraction[5] + Q0_var[5] = 0
 R54: Q0_var[5] + Lost_0_expr_5 = 530.9830759590976
 R55: Q0_var[5] + Q_hat_adjusted_5 = 671.2611178948644
 R56: Sold_0[5] - Q0_var[5] - Q1_var[5] + total_stock_second_period_5 = 0
 R57: total_stock_second_period_5 + Lost_1_expr_5 = 137.96958279269
 R58: Sold_1[5] + Left_1[5] - total_stock_second_period_5 = 0
 R59: - 800 Sold_0[5] - 800 Sold_1[5] + 800 Lost_0[5] + 800 Lost_1[5]
   + 200 Left_1[5] + profits_vars[5] = 0
 Fraction_lower_bound_6: Fraction[6] >= 0
 Fraction_upper_bound_6: Fraction[6] <= 1
 R62: 1.70887173413371 alphas[0] - 1.290830225835101 alphas[1]
   + 0.8486969518898296 alphas[2] + 0.2751000507917559 alphas[3]
   - alphas[4] + f_var[6] = 0
 R63: - 1313.714597627199 Fraction[6] + Q0_var[6] = 0
 R64: Q0_var[6] + Lost_0_expr_6 = 579.1185032127753
 R65: Q0_var[6] + Q_hat_adjusted_6 = 745.3806394611933
 R66: Sold_0[6] - Q0_var[6] - Q1_var[6] + total_stock_second_period_6 = 0
 R67: total_stock_second_period_6 + Lost_1_expr_6 = 164.0034301584347
 R68: Sold_1[6] + Left_1[6] - total_stock_second_period_6 = 0
 R69: - 800 Sold_0[6] - 800 Sold_1[6] + 800 Lost_0[6] + 800 Lost_1[6]
   + 200 Left_1[6] + profits_vars[6] = 0
 Fraction_lower_bound_7: Fraction[7] >= 0
 Fraction_upper_bound_7: Fraction[7] <= 1
 R72: - 0.3503718276298489 alphas[0] + 1.743488500730713 alphas[1]
   - 1.436443650115903 alphas[2] - 0.4945044119160554 alphas[3] - alphas[4]
   + f_var[7] = 0
 R73: - 1313.714597627199 Fraction[7] + Q0_var[7] = 0
 R74: Q0_var[7] + Lost_0_expr_7 = 1179.923877787524
 R75: Q0_var[7] + Q_hat_adjusted_7 = 1474.424217894902
 R76: Sold_0[7] - Q0_var[7] - Q1_var[7] + total_stock_second_period_7 = 0
 R77: total_stock_second_period_7 + Lost_1_expr_7 = 292.3140309082997
 R78: Sold_1[7] + Left_1[7] - total_stock_second_period_7 = 0
 R79: - 800 Sold_0[7] - 800 Sold_1[7] + 800 Lost_0[7] + 800 Lost_1[7]
   + 200 Left_1[7] + profits_vars[7] = 0
 Fraction_lower_bound_8: Fraction[8] >= 0
 Fraction_upper_bound_8: Fraction[8] <= 1
 R82: 1.044356702910634 alphas[0] + 1.373079094824341 alphas[1]
   - 1.230386892644756 alphas[2] - 0.137830936468639 alphas[3] - alphas[4]
   + f_var[8] = 0
 R83: - 1313.714597627199 Fraction[8] + Q0_var[8] = 0
 R84: Q0_var[8] + Lost_0_expr_8 = 1177.346133627161
 R85: Q0_var[8] + Q_hat_adjusted_8 = 1460.162756409497
 R86: Sold_0[8] - Q0_var[8] - Q1_var[8] + total_stock_second_period_8 = 0
 R87: total_stock_second_period_8 + Lost_1_expr_8 = 280.7439479491867
 R88: Sold_1[8] + Left_1[8] - total_stock_second_period_8 = 0
 R89: - 800 Sold_0[8] - 800 Sold_1[8] + 800 Lost_0[8] + 800 Lost_1[8]
   + 200 Left_1[8] + profits_vars[8] = 0
 Fraction_lower_bound_9: Fraction[9] >= 0
 Fraction_upper_bound_9: Fraction[9] <= 1
 R92: 1.134828869508002 alphas[0] + 0.0833468428933028 alphas[1]
   + 1.189570165456632 alphas[2] - 1.647623025233626 alphas[3] - alphas[4]
   + f_var[9] = 0
 R93: - 1313.714597627199 Fraction[9] + Q0_var[9] = 0
 R94: Q0_var[9] + Lost_0_expr_9 = 485.9344047016101
 R95: Q0_var[9] + Q_hat_adjusted_9 = 633.0312900658136
 R96: Sold_0[9] - Q0_var[9] - Q1_var[9] + total_stock_second_period_9 = 0
 R97: total_stock_second_period_9 + Lost_1_expr_9 = 144.863405796243
 R98: Sold_1[9] + Left_1[9] - total_stock_second_period_9 = 0
 R99: - 800 Sold_0[9] - 800 Sold_1[9] + 800 Lost_0[9] + 800 Lost_1[9]
   + 200 Left_1[9] + profits_vars[9] = 0
 Fraction_lower_bound_10: Fraction[10] >= 0
 Fraction_upper_bound_10: Fraction[10] <= 1
 R102: 1.420652956117004 alphas[0] - 0.3567224241454883 alphas[1]
   - 0.6780942011470578 alphas[2] - 0.0444140901928973 alphas[3]
   - alphas[4] + f_var[10] = 0
 R103: - 1313.714597627199 Fraction[10] + Q0_var[10] = 0
 R104: Q0_var[10] + Lost_0_expr_10 = 924.8748576553983
 R105: Q0_var[10] + Q_hat_adjusted_10 = 1176.568891004688
 R106: Sold_0[10] - Q0_var[10] - Q1_var[10] + total_stock_second_period_10
   = 0
 R107: total_stock_second_period_10 + Lost_1_expr_10 = 249.7327239701096
 R108: Sold_1[10] + Left_1[10] - total_stock_second_period_10 = 0
 R109: - 800 Sold_0[10] - 800 Sold_1[10] + 800 Lost_0[10] + 800 Lost_1[10]
   + 200 Left_1[10] + profits_vars[10] = 0
 Fraction_lower_bound_11: Fraction[11] >= 0
 Fraction_upper_bound_11: Fraction[11] <= 1
 R112: 1.315853199607843 alphas[0] + 1.15366834560855 alphas[1]
   - 0.9397296012043972 alphas[2] + 0.8530683123700162 alphas[3]
   - alphas[4] + f_var[11] = 0
 R113: - 1313.714597627199 Fraction[11] + Q0_var[11] = 0
 R114: Q0_var[11] + Lost_0_expr_11 = 960.7120767671063
 R115: Q0_var[11] + Q_hat_adjusted_11 = 1227.347447895563
 R116: Sold_0[11] - Q0_var[11] - Q1_var[11] + total_stock_second_period_11
   = 0
 R117: total_stock_second_period_11 + Lost_1_expr_11 = 264.4235466388491
 R118: Sold_1[11] + Left_1[11] - total_stock_second_period_11 = 0
 R119: - 800 Sold_0[11] - 800 Sold_1[11] + 800 Lost_0[11] + 800 Lost_1[11]
   + 200 Left_1[11] + profits_vars[11] = 0
 Fraction_lower_bound_12: Fraction[12] >= 0
 Fraction_upper_bound_12: Fraction[12] <= 1
 R122: 0.7369869742738262 alphas[0] - 0.6692887477035896 alphas[1]
   - 0.2570230575095985 alphas[2] + 0.1408367694990269 alphas[3]
   - alphas[4] + f_var[12] = 0
 R123: - 1313.714597627199 Fraction[12] + Q0_var[12] = 0
 R124: Q0_var[12] + Lost_0_expr_12 = 848.9068635451672
 R125: Q0_var[12] + Q_hat_adjusted_12 = 1077.070895293197
 R126: Sold_0[12] - Q0_var[12] - Q1_var[12] + total_stock_second_period_12
   = 0
 R127: total_stock_second_period_12 + Lost_1_expr_12 = 226.0895873733634
 R128: Sold_1[12] + Left_1[12] - total_stock_second_period_12 = 0
 R129: - 800 Sold_0[12] - 800 Sold_1[12] + 800 Lost_0[12] + 800 Lost_1[12]
   + 200 Left_1[12] + profits_vars[12] = 0
 Fraction_lower_bound_13: Fraction[13] >= 0
 Fraction_upper_bound_13: Fraction[13] <= 1
 R132: - 1.480159773823298 alphas[0] + 0.7623434406003272 alphas[1]
   + 1.050637445399918 alphas[2] + 1.45683853340321 alphas[3] - alphas[4]
   + f_var[13] = 0
 R133: - 1313.714597627199 Fraction[13] + Q0_var[13] = 0
 R134: Q0_var[13] + Lost_0_expr_13 = 640.6673867048216
 R135: Q0_var[13] + Q_hat_adjusted_13 = 795.491225197135
 R136: Sold_0[13] - Q0_var[13] - Q1_var[13] + total_stock_second_period_13
   = 0
 R137: total_stock_second_period_13 + Lost_1_expr_13 = 152.6644753474297
 R138: Sold_1[13] + Left_1[13] - total_stock_second_period_13 = 0
 R139: - 800 Sold_0[13] - 800 Sold_1[13] + 800 Lost_0[13] + 800 Lost_1[13]
   + 200 Left_1[13] + profits_vars[13] = 0
 Fraction_lower_bound_14: Fraction[14] >= 0
 Fraction_upper_bound_14: Fraction[14] <= 1
 R142: - 1.361022113044733 alphas[0] - 0.2043782642028523 alphas[1]
   + 0.372544718101718 alphas[2] + 1.586018619947777 alphas[3] - alphas[4]
   + f_var[14] = 0
 R143: - 1313.714597627199 Fraction[14] + Q0_var[14] = 0
 R144: Q0_var[14] + Lost_0_expr_14 = 683.4576676850007
 R145: Q0_var[14] + Q_hat_adjusted_14 = 876.3609391604508
 R146: Sold_0[14] - Q0_var[14] - Q1_var[14] + total_stock_second_period_14
   = 0
 R147: total_stock_second_period_14 + Lost_1_expr_14 = 190.7393689112113
 R148: Sold_1[14] + Left_1[14] - total_stock_second_period_14 = 0
 R149: - 800 Sold_0[14] - 800 Sold_1[14] + 800 Lost_0[14] + 800 Lost_1[14]
   + 200 Left_1[14] + profits_vars[14] = 0
 Fraction_lower_bound_15: Fraction[15] >= 0
 Fraction_upper_bound_15: Fraction[15] <= 1
 R152: 0.8482379780171911 alphas[0] - 0.3653614653672564 alphas[1]
   + 1.084050579374082 alphas[2] + 1.129930709656158 alphas[3] - alphas[4]
   + f_var[15] = 0
 R153: - 1313.714597627199 Fraction[15] + Q0_var[15] = 0
 R154: Q0_var[15] + Lost_0_expr_15 = 565.2714624479981
 R155: Q0_var[15] + Q_hat_adjusted_15 = 718.2605572770894
 R156: Sold_0[15] - Q0_var[15] - Q1_var[15] + total_stock_second_period_15
   = 0
 R157: total_stock_second_period_15 + Lost_1_expr_15 = 150.7883285188988
 R158: Sold_1[15] + Left_1[15] - total_stock_second_period_15 = 0
 R159: - 800 Sold_0[15] - 800 Sold_1[15] + 800 Lost_0[15] + 800 Lost_1[15]
   + 200 Left_1[15] + profits_vars[15] = 0
 Fraction_lower_bound_16: Fraction[16] >= 0
 Fraction_upper_bound_16: Fraction[16] <= 1
 R162: 0.1546095964727059 alphas[0] - 1.554590313026028 alphas[1]
   + 1.014236157465197 alphas[2] + 1.579065596860407 alphas[3] - alphas[4]
   + f_var[16] = 0
 R163: - 1313.714597627199 Fraction[16] + Q0_var[16] = 0
 R164: Q0_var[16] + Lost_0_expr_16 = 662.207580975467
 R165: Q0_var[16] + Q_hat_adjusted_16 = 819.1059646121947
 R166: Sold_0[16] - Q0_var[16] - Q1_var[16] + total_stock_second_period_16
   = 0
 R167: total_stock_second_period_16 + Lost_1_expr_16 = 154.7084069550222
 R168: Sold_1[16] + Left_1[16] - total_stock_second_period_16 = 0
 R169: - 800 Sold_0[16] - 800 Sold_1[16] + 800 Lost_0[16] + 800 Lost_1[16]
   + 200 Left_1[16] + profits_vars[16] = 0
 Fraction_lower_bound_17: Fraction[17] >= 0
 Fraction_upper_bound_17: Fraction[17] <= 1
 R172: 1.363487064948847 alphas[0] + 0.317640134850317 alphas[1]
   + 1.327347896359406 alphas[2] + 0.8408055728588897 alphas[3] - alphas[4]
   + f_var[17] = 0
 R173: - 1313.714597627199 Fraction[17] + Q0_var[17] = 0
 R174: Q0_var[17] + Lost_0_expr_17 = 558.377137774725
 R175: Q0_var[17] + Q_hat_adjusted_17 = 697.8560815501468
 R176: Sold_0[17] - Q0_var[17] - Q1_var[17] + total_stock_second_period_17
   = 0
 R177: total_stock_second_period_17 + Lost_1_expr_17 = 137.1271888611869
 R178: Sold_1[17] + Left_1[17] - total_stock_second_period_17 = 0
 R179: - 800 Sold_0[17] - 800 Sold_1[17] + 800 Lost_0[17] + 800 Lost_1[17]
   + 200 Left_1[17] + profits_vars[17] = 0
 Fraction_lower_bound_18: Fraction[18] >= 0
 Fraction_upper_bound_18: Fraction[18] <= 1
 R182: - 0.2221186384768538 alphas[0] + 0.7593696488031217 alphas[1]
   + 1.897154775014547 alphas[2] - 0.7121009931593174 alphas[3] - alphas[4]
   + f_var[18] = 0
 R183: - 1313.714597627199 Fraction[18] + Q0_var[18] = 0
 R184: Q0_var[18] + Lost_0_expr_18 = 415.4275787598831
 R185: Q0_var[18] + Q_hat_adjusted_18 = 522.8763679664962
 R186: Sold_0[18] - Q0_var[18] - Q1_var[18] + total_stock_second_period_18
   = 0
 R187: total_stock_second_period_18 + Lost_1_expr_18 = 105.1325436115658
 R188: Sold_1[18] + Left_1[18] - total_stock_second_period_18 = 0
 R189: - 800 Sold_0[18] - 800 Sold_1[18] + 800 Lost_0[18] + 800 Lost_1[18]
   + 200 Left_1[18] + profits_vars[18] = 0
 Fraction_lower_bound_19: Fraction[19] >= 0
 Fraction_upper_bound_19: Fraction[19] <= 1
 R192: - 1.614240821534305 alphas[0] + 1.023904533599532 alphas[1]
   + 0.0773269264275771 alphas[2] - 0.8142182483791734 alphas[3]
   - alphas[4] + f_var[19] = 0
 R193: - 1313.714597627199 Fraction[19] + Q0_var[19] = 0
 R194: Q0_var[19] + Lost_0_expr_19 = 756.3915232558497
 R195: Q0_var[19] + Q_hat_adjusted_19 = 965.8982849549175
 R196: Sold_0[19] - Q0_var[19] - Q1_var[19] + total_stock_second_period_19
   = 0
 R197: total_stock_second_period_19 + Lost_1_expr_19 = 207.3158422217444
 R198: Sold_1[19] + Left_1[19] - total_stock_second_period_19 = 0
 R199: - 800 Sold_0[19] - 800 Sold_1[19] + 800 Lost_0[19] + 800 Lost_1[19]
   + 200 Left_1[19] + profits_vars[19] = 0
 Fraction_lower_bound_20: Fraction[20] >= 0
 Fraction_upper_bound_20: Fraction[20] <= 1
 R202: 1.567954033251472 alphas[0] - 1.328892497584955 alphas[1]
   + 1.518150590410068 alphas[2] + 1.617495612276112 alphas[3] - alphas[4]
   + f_var[20] = 0
 R203: - 1313.714597627199 Fraction[20] + Q0_var[20] = 0
 R204: Q0_var[20] + Lost_0_expr_20 = 488.4966268091613
 R205: Q0_var[20] + Q_hat_adjusted_20 = 617.0208699188074
 R206: Sold_0[20] - Q0_var[20] - Q1_var[20] + total_stock_second_period_20
   = 0
 R207: total_stock_second_period_20 + Lost_1_expr_20 = 126.4136212633655
 R208: Sold_1[20] + Left_1[20] - total_stock_second_period_20 = 0
 R209: - 800 Sold_0[20] - 800 Sold_1[20] + 800 Lost_0[20] + 800 Lost_1[20]
   + 200 Left_1[20] + profits_vars[20] = 0
 Fraction_lower_bound_21: Fraction[21] >= 0
 Fraction_upper_bound_21: Fraction[21] <= 1
 R212: - 0.5317151746282208 alphas[0] - 0.7313455181264898 alphas[1]
   + 1.930042011465057 alphas[2] - 0.1650713318874335 alphas[3] - alphas[4]
   + f_var[21] = 0
 R213: - 1313.714597627199 Fraction[21] + Q0_var[21] = 0
 R214: Q0_var[21] + Lost_0_expr_21 = 433.7344164331509
 R215: Q0_var[21] + Q_hat_adjusted_21 = 539.4349584759725
 R216: Sold_0[21] - Q0_var[21] - Q1_var[21] + total_stock_second_period_21
   = 0
 R217: total_stock_second_period_21 + Lost_1_expr_21 = 103.285925985933
 R218: Sold_1[21] + Left_1[21] - total_stock_second_period_21 = 0
 R219: - 800 Sold_0[21] - 800 Sold_1[21] + 800 Lost_0[21] + 800 Lost_1[21]
   + 200 Left_1[21] + profits_vars[21] = 0
 Fraction_lower_bound_22: Fraction[22] >= 0
 Fraction_upper_bound_22: Fraction[22] <= 1
 R222: - 0.1391653146474336 alphas[0] - 1.576065670454613 alphas[1]
   + 1.348834161070553 alphas[2] + 1.487196992586131 alphas[3] - alphas[4]
   + f_var[22] = 0
 R223: - 1313.714597627199 Fraction[22] + Q0_var[22] = 0
 R224: Q0_var[22] + Lost_0_expr_22 = 566.1747417586032
 R225: Q0_var[22] + Q_hat_adjusted_22 = 704.3480477896568
 R226: Sold_0[22] - Q0_var[22] - Q1_var[22] + total_stock_second_period_22
   = 0
 R227: total_stock_second_period_22 + Lost_1_expr_22 = 135.9207354701485
 R228: Sold_1[22] + Left_1[22] - total_stock_second_period_22 = 0
 R229: - 800 Sold_0[22] - 800 Sold_1[22] + 800 Lost_0[22] + 800 Lost_1[22]
   + 200 Left_1[22] + profits_vars[22] = 0
 Fraction_lower_bound_23: Fraction[23] >= 0
 Fraction_upper_bound_23: Fraction[23] <= 1
 R232: 0.1293480861249456 alphas[0] + 1.617497205919207 alphas[1]
   - 1.133451883654001 alphas[2] + 0.6065469705721442 alphas[3] - alphas[4]
   + f_var[23] = 0
 R233: - 1313.714597627199 Fraction[23] + Q0_var[23] = 0
 R234: Q0_var[23] + Lost_0_expr_23 = 1113.853236533312
 R235: Q0_var[23] + Q_hat_adjusted_23 = 1391.04544997757
 R236: Sold_0[23] - Q0_var[23] - Q1_var[23] + total_stock_second_period_23
   = 0
 R237: total_stock_second_period_23 + Lost_1_expr_23 = 275.3010490633181
 R238: Sold_1[23] + Left_1[23] - total_stock_second_period_23 = 0
 R239: - 800 Sold_0[23] - 800 Sold_1[23] + 800 Lost_0[23] + 800 Lost_1[23]
   + 200 Left_1[23] + profits_vars[23] = 0
 Fraction_lower_bound_24: Fraction[24] >= 0
 Fraction_upper_bound_24: Fraction[24] <= 1
 R242: - 0.5821135100126386 alphas[0] - 0.667439066546644 alphas[1]
   + 0.5778418206483907 alphas[2] + 1.501998139189441 alphas[3] - alphas[4]
   + f_var[24] = 0
 R243: - 1313.714597627199 Fraction[24] + Q0_var[24] = 0
 R244: Q0_var[24] + Lost_0_expr_24 = 674.7920442704966
 R245: Q0_var[24] + Q_hat_adjusted_24 = 856.1740420310929
 R246: Sold_0[24] - Q0_var[24] - Q1_var[24] + total_stock_second_period_24
   = 0
 R247: total_stock_second_period_24 + Lost_1_expr_24 = 179.2119405613942
 R248: Sold_1[24] + Left_1[24] - total_stock_second_period_24 = 0
 R249: - 800 Sold_0[24] - 800 Sold_1[24] + 800 Lost_0[24] + 800 Lost_1[24]
   + 200 Left_1[24] + profits_vars[24] = 0
 Fraction_lower_bound_25: Fraction[25] >= 0
 Fraction_upper_bound_25: Fraction[25] <= 1
 R252: 0.6690479338220655 alphas[0] - 0.7038570092006652 alphas[1]
   + 1.472630884670251 alphas[2] - 0.2855425336265254 alphas[3] - alphas[4]
   + f_var[25] = 0
 R253: - 1313.714597627199 Fraction[25] + Q0_var[25] = 0
 R254: Q0_var[25] + Lost_0_expr_25 = 542.0494353456584
 R255: Q0_var[25] + Q_hat_adjusted_25 = 673.2275130198666
 R256: Sold_0[25] - Q0_var[25] - Q1_var[25] + total_stock_second_period_25
   = 0
 R257: total_stock_second_period_25 + Lost_1_expr_25 = 128.9695519040114
 R258: Sold_1[25] + Left_1[25] - total_stock_second_period_25 = 0
 R259: - 800 Sold_0[25] - 800 Sold_1[25] + 800 Lost_0[25] + 800 Lost_1[25]
   + 200 Left_1[25] + profits_vars[25] = 0
 Fraction_lower_bound_26: Fraction[26] >= 0
 Fraction_upper_bound_26: Fraction[26] <= 1
 R262: 0.2387653877903883 alphas[0] - 0.273994006823304 alphas[1]
   - 0.4313591338703168 alphas[2] + 1.303702227523297 alphas[3] - alphas[4]
   + f_var[26] = 0
 R263: - 1313.714597627199 Fraction[26] + Q0_var[26] = 0
 R264: Q0_var[26] + Lost_0_expr_26 = 970.052225305534
 R265: Q0_var[26] + Q_hat_adjusted_26 = 1207.979576051514
 R266: Sold_0[26] - Q0_var[26] - Q1_var[26] + total_stock_second_period_26
   = 0
 R267: total_stock_second_period_26 + Lost_1_expr_26 = 235.8785546997134
 R268: Sold_1[26] + Left_1[26] - total_stock_second_period_26 = 0
 R269: - 800 Sold_0[26] - 800 Sold_1[26] + 800 Lost_0[26] + 800 Lost_1[26]
   + 200 Left_1[26] + profits_vars[26] = 0
 Fraction_lower_bound_27: Fraction[27] >= 0
 Fraction_upper_bound_27: Fraction[27] <= 1
 R272: 1.763548423633227 alphas[0] + 0.2804874779827878 alphas[1]
   - 1.433823893282242 alphas[2] + 1.115356598631139 alphas[3] - alphas[4]
   + f_var[27] = 0
 R273: - 1313.714597627199 Fraction[27] + Q0_var[27] = 0
 R274: Q0_var[27] + Lost_0_expr_27 = 1212.708785929688
 R275: Q0_var[27] + Q_hat_adjusted_27 = 1506.843297813592
 R276: Sold_0[27] - Q0_var[27] - Q1_var[27] + total_stock_second_period_27
   = 0
 R277: total_stock_second_period_27 + Lost_1_expr_27 = 292.1669316126
 R278: Sold_1[27] + Left_1[27] - total_stock_second_period_27 = 0
 R279: - 800 Sold_0[27] - 800 Sold_1[27] + 800 Lost_0[27] + 800 Lost_1[27]
   + 200 Left_1[27] + profits_vars[27] = 0
 Fraction_lower_bound_28: Fraction[28] >= 0
 Fraction_upper_bound_28: Fraction[28] <= 1
 R282: 1.554139408776133 alphas[0] - 0.2218848343771625 alphas[1]
   - 0.2122456781117908 alphas[2] - 1.622240245118781 alphas[3] - alphas[4]
   + f_var[28] = 0
 R283: - 1313.714597627199 Fraction[28] + Q0_var[28] = 0
 R284: Q0_var[28] + Lost_0_expr_28 = 937.7203361721765
 R285: Q0_var[28] + Q_hat_adjusted_28 = 1163.457307271965
 R286: Sold_0[28] - Q0_var[28] - Q1_var[28] + total_stock_second_period_28
   = 0
 R287: total_stock_second_period_28 + Lost_1_expr_28 = 223.5753383834952
 R288: Sold_1[28] + Left_1[28] - total_stock_second_period_28 = 0
 R289: - 800 Sold_0[28] - 800 Sold_1[28] + 800 Lost_0[28] + 800 Lost_1[28]
   + 200 Left_1[28] + profits_vars[28] = 0
 Fraction_lower_bound_29: Fraction[29] >= 0
 Fraction_upper_bound_29: Fraction[29] <= 1
 R292: 0.7839508888305918 alphas[0] + 0.5937928996421402 alphas[1]
   - 1.364520384688983 alphas[2] - 1.067027131009432 alphas[3] - alphas[4]
   + f_var[29] = 0
 R293: - 1313.714597627199 Fraction[29] + Q0_var[29] = 0
 R294: Q0_var[29] + Lost_0_expr_29 = 1102.630414813016
 R295: Q0_var[29] + Q_hat_adjusted_29 = 1392.818359364971
 R296: Sold_0[29] - Q0_var[29] - Q1_var[29] + total_stock_second_period_29
   = 0
 R297: total_stock_second_period_29 + Lost_1_expr_29 = 288.2755409412997
 R298: Sold_1[29] + Left_1[29] - total_stock_second_period_29 = 0
 R299: - 800 Sold_0[29] - 800 Sold_1[29] + 800 Lost_0[29] + 800 Lost_1[29]
   + 200 Left_1[29] + profits_vars[29] = 0
 Fraction_lower_bound_30: Fraction[30] >= 0
 Fraction_upper_bound_30: Fraction[30] <= 1
 R302: 1.220145713770978 alphas[0] - 0.6447152707474093 alphas[1]
   + 1.130757141020475 alphas[2] + 1.108577021370274 alphas[3] - alphas[4]
   + f_var[30] = 0
 R303: - 1313.714597627199 Fraction[30] + Q0_var[30] = 0
 R304: Q0_var[30] + Lost_0_expr_30 = 644.085155158248
 R305: Q0_var[30] + Q_hat_adjusted_30 = 794.4478828617855
 R306: Sold_0[30] - Q0_var[30] - Q1_var[30] + total_stock_second_period_30
   = 0
 R307: total_stock_second_period_30 + Lost_1_expr_30 = 148.1657559812276
 R308: Sold_1[30] + Left_1[30] - total_stock_second_period_30 = 0
 R309: - 800 Sold_0[30] - 800 Sold_1[30] + 800 Lost_0[30] + 800 Lost_1[30]
   + 200 Left_1[30] + profits_vars[30] = 0
 Fraction_lower_bound_31: Fraction[31] >= 0
 Fraction_upper_bound_31: Fraction[31] <= 1
 R312: 0.6600490114712003 alphas[0] + 1.229295250014619 alphas[1]
   + 0.9448567936877085 alphas[2] - 1.286034155576843 alphas[3] - alphas[4]
   + f_var[31] = 0
 R313: - 1313.714597627199 Fraction[31] + Q0_var[31] = 0
 R314: Q0_var[31] + Lost_0_expr_31 = 619.6463570481958
 R315: Q0_var[31] + Q_hat_adjusted_31 = 780.506763091613
 R316: Sold_0[31] - Q0_var[31] - Q1_var[31] + total_stock_second_period_31
   = 0
 R317: total_stock_second_period_31 + Lost_1_expr_31 = 158.6040569015613
 R318: Sold_1[31] + Left_1[31] - total_stock_second_period_31 = 0
 R319: - 800 Sold_0[31] - 800 Sold_1[31] + 800 Lost_0[31] + 800 Lost_1[31]
   + 200 Left_1[31] + profits_vars[31] = 0
 Fraction_lower_bound_32: Fraction[32] >= 0
 Fraction_upper_bound_32: Fraction[32] <= 1
 R322: 0.6579857315599946 alphas[0] + 0.407813287848192 alphas[1]
   + 0.6757050659270109 alphas[2] + 1.552860131025213 alphas[3] - alphas[4]
   + f_var[32] = 0
 R323: - 1313.714597627199 Fraction[32] + Q0_var[32] = 0
 R324: Q0_var[32] + Lost_0_expr_32 = 687.618944322122
 R325: Q0_var[32] + Q_hat_adjusted_32 = 863.5205376860163
 R326: Sold_0[32] - Q0_var[32] - Q1_var[32] + total_stock_second_period_32
   = 0
 R327: total_stock_second_period_32 + Lost_1_expr_32 = 173.7169212259221
 R328: Sold_1[32] + Left_1[32] - total_stock_second_period_32 = 0
 R329: - 800 Sold_0[32] - 800 Sold_1[32] + 800 Lost_0[32] + 800 Lost_1[32]
   + 200 Left_1[32] + profits_vars[32] = 0
 Fraction_lower_bound_33: Fraction[33] >= 0
 Fraction_upper_bound_33: Fraction[33] <= 1
 R332: - 1.171422365946754 alphas[0] - 0.7592584856575121 alphas[1]
   + 1.760760968370906 alphas[2] - 1.451444148745465 alphas[3] - alphas[4]
   + f_var[33] = 0
 R333: - 1313.714597627199 Fraction[33] + Q0_var[33] = 0
 R334: Q0_var[33] + Lost_0_expr_33 = 404.1361694017194
 R335: Q0_var[33] + Q_hat_adjusted_33 = 519.1190653583624
 R336: Sold_0[33] - Q0_var[33] - Q1_var[33] + total_stock_second_period_33
   = 0
 R337: total_stock_second_period_33 + Lost_1_expr_33 = 112.7910533381431
 R338: Sold_1[33] + Left_1[33] - total_stock_second_period_33 = 0
 R339: - 800 Sold_0[33] - 800 Sold_1[33] + 800 Lost_0[33] + 800 Lost_1[33]
   + 200 Left_1[33] + profits_vars[33] = 0
 Fraction_lower_bound_34: Fraction[34] >= 0
 Fraction_upper_bound_34: Fraction[34] <= 1
 R342: 0.2351926912893569 alphas[0] - 1.245180298746146 alphas[1]
   - 1.361151694471933 alphas[2] + 1.650525889233109 alphas[3] - alphas[4]
   + f_var[34] = 0
 R343: - 1313.714597627199 Fraction[34] + Q0_var[34] = 0
 R344: Q0_var[34] + Lost_0_expr_34 = 1188.151607817763
 R345: Q0_var[34] + Q_hat_adjusted_34 = 1478.352413186868
 R346: Sold_0[34] - Q0_var[34] - Q1_var[34] + total_stock_second_period_34
   = 0
 R347: total_stock_second_period_34 + Lost_1_expr_34 = 288.0863890505626
 R348: Sold_1[34] + Left_1[34] - total_stock_second_period_34 = 0
 R349: - 800 Sold_0[34] - 800 Sold_1[34] + 800 Lost_0[34] + 800 Lost_1[34]
   + 200 Left_1[34] + profits_vars[34] = 0
 Fraction_lower_bound_35: Fraction[35] >= 0
 Fraction_upper_bound_35: Fraction[35] <= 1
 R352: - 1.54733843478505 alphas[0] + 0.7477875926475565 alphas[1]
   + 1.22094809915602 alphas[2] - 0.7072024565632828 alphas[3] - alphas[4]
   + f_var[35] = 0
 R353: - 1313.714597627199 Fraction[35] + Q0_var[35] = 0
 R354: Q0_var[35] + Lost_0_expr_35 = 519.7215275568076
 R355: Q0_var[35] + Q_hat_adjusted_35 = 665.1495446644216
 R356: Sold_0[35] - Q0_var[35] - Q1_var[35] + total_stock_second_period_35
   = 0
 R357: total_stock_second_period_35 + Lost_1_expr_35 = 143.101535422734
 R358: Sold_1[35] + Left_1[35] - total_stock_second_period_35 = 0
 R359: - 800 Sold_0[35] - 800 Sold_1[35] + 800 Lost_0[35] + 800 Lost_1[35]
   + 200 Left_1[35] + profits_vars[35] = 0
 Fraction_lower_bound_36: Fraction[36] >= 0
 Fraction_upper_bound_36: Fraction[36] <= 1
 R362: - 0.9608293712909705 alphas[0] - 1.293054003572672 alphas[1]
   - 0.1069577403950463 alphas[2] - 1.296134531333326 alphas[3] - alphas[4]
   + f_var[36] = 0
 R363: - 1313.714597627199 Fraction[36] + Q0_var[36] = 0
 R364: Q0_var[36] + Lost_0_expr_36 = 896.7886466557181
 R365: Q0_var[36] + Q_hat_adjusted_36 = 1116.59125947528
 R366: Sold_0[36] - Q0_var[36] - Q1_var[36] + total_stock_second_period_36
   = 0
 R367: total_stock_second_period_36 + Lost_1_expr_36 = 217.6634227108934
 R368: Sold_1[36] + Left_1[36] - total_stock_second_period_36 = 0
 R369: - 800 Sold_0[36] - 800 Sold_1[36] + 800 Lost_0[36] + 800 Lost_1[36]
   + 200 Left_1[36] + profits_vars[36] = 0
 Fraction_lower_bound_37: Fraction[37] >= 0
 Fraction_upper_bound_37: Fraction[37] <= 1
 R372: - 0.5203688533588062 alphas[0] - 1.031278031024805 alphas[1]
   + 0.0217142312364571 alphas[2] + 0.7855354799590643 alphas[3]
   - alphas[4] + f_var[37] = 0
 R373: - 1313.714597627199 Fraction[37] + Q0_var[37] = 0
 R374: Q0_var[37] + Lost_0_expr_37 = 823.1992212479963
 R375: Q0_var[37] + Q_hat_adjusted_37 = 1035.800908262489
 R376: Sold_0[37] - Q0_var[37] - Q1_var[37] + total_stock_second_period_37
   = 0
 R377: total_stock_second_period_37 + Lost_1_expr_37 = 210.4384939845112
 R378: Sold_1[37] + Left_1[37] - total_stock_second_period_37 = 0
 R379: - 800 Sold_0[37] - 800 Sold_1[37] + 800 Lost_0[37] + 800 Lost_1[37]
   + 200 Left_1[37] + profits_vars[37] = 0
 Fraction_lower_bound_38: Fraction[38] >= 0
 Fraction_upper_bound_38: Fraction[38] <= 1
 R382: 1.293362875779139 alphas[0] - 1.239046342061042 alphas[1]
   - 1.32231774182963 alphas[2] + 0.1452197032167309 alphas[3] - alphas[4]
   + f_var[38] = 0
 R383: - 1313.714597627199 Fraction[38] + Q0_var[38] = 0
 R384: Q0_var[38] + Lost_0_expr_38 = 1115.579999030597
 R385: Q0_var[38] + Q_hat_adjusted_38 = 1403.31449472446
 R386: Sold_0[38] - Q0_var[38] - Q1_var[38] + total_stock_second_period_38
   = 0
 R387: total_stock_second_period_38 + Lost_1_expr_38 = 285.9058633584382
 R388: Sold_1[38] + Left_1[38] - total_stock_second_period_38 = 0
 R389: - 800 Sold_0[38] - 800 Sold_1[38] + 800 Lost_0[38] + 800 Lost_1[38]
   + 200 Left_1[38] + profits_vars[38] = 0
 Fraction_lower_bound_39: Fraction[39] >= 0
 Fraction_upper_bound_39: Fraction[39] <= 1
 R392: - 0.8238351914822081 alphas[0] - 0.9833622863609263 alphas[1]
   + 1.163267048690253 alphas[2] + 0.9197147159080235 alphas[3] - alphas[4]
   + f_var[39] = 0
 R393: - 1313.714597627199 Fraction[39] + Q0_var[39] = 0
 R394: Q0_var[39] + Lost_0_expr_39 = 505.6800124242206
 R395: Q0_var[39] + Q_hat_adjusted_39 = 654.2323883801105
 R396: Sold_0[39] - Q0_var[39] - Q1_var[39] + total_stock_second_period_39
   = 0
 R397: total_stock_second_period_39 + Lost_1_expr_39 = 146.3403252941073
 R398: Sold_1[39] + Left_1[39] - total_stock_second_period_39 = 0
 R399: - 800 Sold_0[39] - 800 Sold_1[39] + 800 Lost_0[39] + 800 Lost_1[39]
   + 200 Left_1[39] + profits_vars[39] = 0
 Fraction_lower_bound_40: Fraction[40] >= 0
 Fraction_upper_bound_40: Fraction[40] <= 1
 R402: - 0.7390381194910232 alphas[0] + 0.858830868126074 alphas[1]
   - 0.4917357006997582 alphas[2] - 0.9872385986521026 alphas[3]
   - alphas[4] + f_var[40] = 0
 R403: - 1313.714597627199 Fraction[40] + Q0_var[40] = 0
 R404: Q0_var[40] + Lost_0_expr_40 = 890.4839503940082
 R405: Q0_var[40] + Q_hat_adjusted_40 = 1132.044845137017
 R406: Sold_0[40] - Q0_var[40] - Q1_var[40] + total_stock_second_period_40
   = 0
 R407: total_stock_second_period_40 + Lost_1_expr_40 = 239.2686977630919
 R408: Sold_1[40] + Left_1[40] - total_stock_second_period_40 = 0
 R409: - 800 Sold_0[40] - 800 Sold_1[40] + 800 Lost_0[40] + 800 Lost_1[40]
   + 200 Left_1[40] + profits_vars[40] = 0
 Fraction_lower_bound_41: Fraction[41] >= 0
 Fraction_upper_bound_41: Fraction[41] <= 1
 R412: 1.391310870157227 alphas[0] + 0.2132073892791626 alphas[1]
   + 1.631326636679523 alphas[2] - 1.451228331231364 alphas[3] - alphas[4]
   + f_var[41] = 0
 R413: - 1313.714597627199 Fraction[41] + Q0_var[41] = 0
 R414: Q0_var[41] + Lost_0_expr_41 = 479.5083968544993
 R415: Q0_var[41] + Q_hat_adjusted_41 = 601.8563775579182
 R416: Sold_0[41] - Q0_var[41] - Q1_var[41] + total_stock_second_period_41
   = 0
 R417: total_stock_second_period_41 + Lost_1_expr_41 = 120.0587884463515
 R418: Sold_1[41] + Left_1[41] - total_stock_second_period_41 = 0
 R419: - 800 Sold_0[41] - 800 Sold_1[41] + 800 Lost_0[41] + 800 Lost_1[41]
   + 200 Left_1[41] + profits_vars[41] = 0
 Fraction_lower_bound_42: Fraction[42] >= 0
 Fraction_upper_bound_42: Fraction[42] <= 1
 R422: - 0.6771926318868985 alphas[0] + 0.275392767024887 alphas[1]
   + 0.1457155659555137 alphas[2] + 0.1347053450344267 alphas[3]
   - alphas[4] + f_var[42] = 0
 R423: - 1313.714597627199 Fraction[42] + Q0_var[42] = 0
 R424: Q0_var[42] + Lost_0_expr_42 = 926.2854918588041
 R425: Q0_var[42] + Q_hat_adjusted_42 = 1131.906867607329
 R426: Sold_0[42] - Q0_var[42] - Q1_var[42] + total_stock_second_period_42
   = 0
 R427: total_stock_second_period_42 + Lost_1_expr_42 = 203.475821430823
 R428: Sold_1[42] + Left_1[42] - total_stock_second_period_42 = 0
 R429: - 800 Sold_0[42] - 800 Sold_1[42] + 800 Lost_0[42] + 800 Lost_1[42]
   + 200 Left_1[42] + profits_vars[42] = 0
 Fraction_lower_bound_43: Fraction[43] >= 0
 Fraction_upper_bound_43: Fraction[43] <= 1
 R432: 0.240721182027268 alphas[0] + 0.5156637013297165 alphas[1]
   + 1.569927101094035 alphas[2] + 1.073141666293822 alphas[3] - alphas[4]
   + f_var[43] = 0
 R433: - 1313.714597627199 Fraction[43] + Q0_var[43] = 0
 R434: Q0_var[43] + Lost_0_expr_43 = 478.4658654171332
 R435: Q0_var[43] + Q_hat_adjusted_43 = 604.2819441188226
 R436: Sold_0[43] - Q0_var[43] - Q1_var[43] + total_stock_second_period_43
   = 0
 R437: total_stock_second_period_43 + Lost_1_expr_43 = 123.5063711923435
 R438: Sold_1[43] + Left_1[43] - total_stock_second_period_43 = 0
 R439: - 800 Sold_0[43] - 800 Sold_1[43] + 800 Lost_0[43] + 800 Lost_1[43]
   + 200 Left_1[43] + profits_vars[43] = 0
 Fraction_lower_bound_44: Fraction[44] >= 0
 Fraction_upper_bound_44: Fraction[44] <= 1
 R442: 0.2878397951507509 alphas[0] + 0.933599379835423 alphas[1]
   + 1.900659636995238 alphas[2] + 0.3225868359527828 alphas[3] - alphas[4]
   + f_var[44] = 0
 R443: - 1313.714597627199 Fraction[44] + Q0_var[44] = 0
 R444: Q0_var[44] + Lost_0_expr_44 = 559.628089659188
 R445: Q0_var[44] + Q_hat_adjusted_44 = 666.8323356160922
 R446: Sold_0[44] - Q0_var[44] - Q1_var[44] + total_stock_second_period_44
   = 0
 R447: total_stock_second_period_44 + Lost_1_expr_44 = 104.935745678194
 R448: Sold_1[44] + Left_1[44] - total_stock_second_period_44 = 0
 R449: - 800 Sold_0[44] - 800 Sold_1[44] + 800 Lost_0[44] + 800 Lost_1[44]
   + 200 Left_1[44] + profits_vars[44] = 0
 Fraction_lower_bound_45: Fraction[45] >= 0
 Fraction_upper_bound_45: Fraction[45] <= 1
 R452: 1.473190152769477 alphas[0] + 0.4138473058972331 alphas[1]
   - 0.7126980740642664 alphas[2] + 0.9006213796205483 alphas[3]
   - alphas[4] + f_var[45] = 0
 R453: - 1313.714597627199 Fraction[45] + Q0_var[45] = 0
 R454: Q0_var[45] + Lost_0_expr_45 = 1010.635998079604
 R455: Q0_var[45] + Q_hat_adjusted_45 = 1264.419408514016
 R456: Sold_0[45] - Q0_var[45] - Q1_var[45] + total_stock_second_period_45
   = 0
 R457: total_stock_second_period_45 + Lost_1_expr_45 = 251.6757307672283
 R458: Sold_1[45] + Left_1[45] - total_stock_second_period_45 = 0
 R459: - 800 Sold_0[45] - 800 Sold_1[45] + 800 Lost_0[45] + 800 Lost_1[45]
   + 200 Left_1[45] + profits_vars[45] = 0
 Fraction_lower_bound_46: Fraction[46] >= 0
 Fraction_upper_bound_46: Fraction[46] <= 1
 R462: - 1.152665795955467 alphas[0] - 0.2361114879256893 alphas[1]
   + 0.6227066931701989 alphas[2] - 0.2135027823518049 alphas[3]
   - alphas[4] + f_var[46] = 0
 R463: - 1313.714597627199 Fraction[46] + Q0_var[46] = 0
 R464: Q0_var[46] + Lost_0_expr_46 = 656.7796503477273
 R465: Q0_var[46] + Q_hat_adjusted_46 = 835.6330755157364
 R466: Sold_0[46] - Q0_var[46] - Q1_var[46] + total_stock_second_period_46
   = 0
 R467: total_stock_second_period_46 + Lost_1_expr_46 = 176.6927788344125
 R468: Sold_1[46] + Left_1[46] - total_stock_second_period_46 = 0
 R469: - 800 Sold_0[46] - 800 Sold_1[46] + 800 Lost_0[46] + 800 Lost_1[46]
   + 200 Left_1[46] + profits_vars[46] = 0
 Fraction_lower_bound_47: Fraction[47] >= 0
 Fraction_upper_bound_47: Fraction[47] <= 1
 R472: 0.5137856728726525 alphas[0] - 1.345155758305791 alphas[1]
   - 1.088395271946271 alphas[2] + 0.4525258076950618 alphas[3] - alphas[4]
   + f_var[47] = 0
 R473: - 1313.714597627199 Fraction[47] + Q0_var[47] = 0
 R474: Q0_var[47] + Lost_0_expr_47 = 1080.132772155145
 R475: Q0_var[47] + Q_hat_adjusted_47 = 1354.893333242068
 R476: Sold_0[47] - Q0_var[47] - Q1_var[47] + total_stock_second_period_47
   = 0
 R477: total_stock_second_period_47 + Lost_1_expr_47 = 272.7711211846461
 R478: Sold_1[47] + Left_1[47] - total_stock_second_period_47 = 0
 R479: - 800 Sold_0[47] - 800 Sold_1[47] + 800 Lost_0[47] + 800 Lost_1[47]
   + 200 Left_1[47] + profits_vars[47] = 0
 Fraction_lower_bound_48: Fraction[48] >= 0
 Fraction_upper_bound_48: Fraction[48] <= 1
 R482: - 1.544094947180397 alphas[0] - 0.4852729009550245 alphas[1]
   + 0.9996798871807365 alphas[2] - 0.3214538813056343 alphas[3]
   - alphas[4] + f_var[48] = 0
 R483: - 1313.714597627199 Fraction[48] + Q0_var[48] = 0
 R484: Q0_var[48] + Lost_0_expr_48 = 721.7531147365655
 R485: Q0_var[48] + Q_hat_adjusted_48 = 879.4636312307366
 R486: Sold_0[48] - Q0_var[48] - Q1_var[48] + total_stock_second_period_48
   = 0
 R487: total_stock_second_period_48 + Lost_1_expr_48 = 155.525741258146
 R488: Sold_1[48] + Left_1[48] - total_stock_second_period_48 = 0
 R489: - 800 Sold_0[48] - 800 Sold_1[48] + 800 Lost_0[48] + 800 Lost_1[48]
   + 200 Left_1[48] + profits_vars[48] = 0
 Fraction_lower_bound_49: Fraction[49] >= 0
 Fraction_upper_bound_49: Fraction[49] <= 1
 R492: - 0.4949580917493536 alphas[0] - 0.2181164634981861 alphas[1]
   + 1.749026804618549 alphas[2] + 0.947518154810715 alphas[3] - alphas[4]
   + f_var[49] = 0
 R493: - 1313.714597627199 Fraction[49] + Q0_var[49] = 0
 R494: Q0_var[49] + Lost_0_expr_49 = 458.6193644477646
 R495: Q0_var[49] + Q_hat_adjusted_49 = 574.3243878805382
 R496: Sold_0[49] - Q0_var[49] - Q1_var[49] + total_stock_second_period_49
   = 0
 R497: total_stock_second_period_49 + Lost_1_expr_49 = 113.4499262923876
 R498: Sold_1[49] + Left_1[49] - total_stock_second_period_49 = 0
 R499: - 800 Sold_0[49] - 800 Sold_1[49] + 800 Lost_0[49] + 800 Lost_1[49]
   + 200 Left_1[49] + profits_vars[49] = 0
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
 Q0_var[0] <= 1314.714597627199
 Q0_var[1] <= 1314.714597627199
 Q0_var[2] <= 1314.714597627199
 Q0_var[3] <= 1314.714597627199
 Q0_var[4] <= 1314.714597627199
 Q0_var[5] <= 1314.714597627199
 Q0_var[6] <= 1314.714597627199
 Q0_var[7] <= 1314.714597627199
 Q0_var[8] <= 1314.714597627199
 Q0_var[9] <= 1314.714597627199
 Q0_var[10] <= 1314.714597627199
 Q0_var[11] <= 1314.714597627199
 Q0_var[12] <= 1314.714597627199
 Q0_var[13] <= 1314.714597627199
 Q0_var[14] <= 1314.714597627199
 Q0_var[15] <= 1314.714597627199
 Q0_var[16] <= 1314.714597627199
 Q0_var[17] <= 1314.714597627199
 Q0_var[18] <= 1314.714597627199
 Q0_var[19] <= 1314.714597627199
 Q0_var[20] <= 1314.714597627199
 Q0_var[21] <= 1314.714597627199
 Q0_var[22] <= 1314.714597627199
 Q0_var[23] <= 1314.714597627199
 Q0_var[24] <= 1314.714597627199
 Q0_var[25] <= 1314.714597627199
 Q0_var[26] <= 1314.714597627199
 Q0_var[27] <= 1314.714597627199
 Q0_var[28] <= 1314.714597627199
 Q0_var[29] <= 1314.714597627199
 Q0_var[30] <= 1314.714597627199
 Q0_var[31] <= 1314.714597627199
 Q0_var[32] <= 1314.714597627199
 Q0_var[33] <= 1314.714597627199
 Q0_var[34] <= 1314.714597627199
 Q0_var[35] <= 1314.714597627199
 Q0_var[36] <= 1314.714597627199
 Q0_var[37] <= 1314.714597627199
 Q0_var[38] <= 1314.714597627199
 Q0_var[39] <= 1314.714597627199
 Q0_var[40] <= 1314.714597627199
 Q0_var[41] <= 1314.714597627199
 Q0_var[42] <= 1314.714597627199
 Q0_var[43] <= 1314.714597627199
 Q0_var[44] <= 1314.714597627199
 Q0_var[45] <= 1314.714597627199
 Q0_var[46] <= 1314.714597627199
 Q0_var[47] <= 1314.714597627199
 Q0_var[48] <= 1314.714597627199
 Q0_var[49] <= 1314.714597627199
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
 min_sales_constr_0: Sold_0[0] = MIN ( Q0_var[0] , 998.441375369316 )
 max_lost_constr_0: Lost_0[0] = MAX ( Lost_0_expr_0 , 0 )
 max_Q1_constr_0: Q1_var[0] = MAX ( Q_hat_adjusted_0 , 0 )
 min_sales2_constr_0: Sold_1[0] = MIN ( total_stock_second_period_0
   , 237.4976552775631 )
 max_lost2_constr_0: Lost_1[0] = MAX ( Lost_1_expr_0 , 0 )
 GC6: Fraction[1] = LOGISTIC ( f_var[1] )
 min_sales_constr_1: Sold_0[1] = MIN ( Q0_var[1] , 1185.759218896487 )
 max_lost_constr_1: Lost_0[1] = MAX ( Lost_0_expr_1 , 0 )
 max_Q1_constr_1: Q1_var[1] = MAX ( Q_hat_adjusted_1 , 0 )
 min_sales2_constr_1: Sold_1[1] = MIN ( total_stock_second_period_1
   , 296.3658779636506 )
 max_lost2_constr_1: Lost_1[1] = MAX ( Lost_1_expr_1 , 0 )
 GC12: Fraction[2] = LOGISTIC ( f_var[2] )
 min_sales_constr_2: Sold_0[2] = MIN ( Q0_var[2] , 610.2814917716669 )
 max_lost_constr_2: Lost_0[2] = MAX ( Lost_0_expr_2 , 0 )
 max_Q1_constr_2: Q1_var[2] = MAX ( Q_hat_adjusted_2 , 0 )
 min_sales2_constr_2: Sold_1[2] = MIN ( total_stock_second_period_2
   , 175.5503678586944 )
 max_lost2_constr_2: Lost_1[2] = MAX ( Lost_1_expr_2 , 0 )
 GC18: Fraction[3] = LOGISTIC ( f_var[3] )
 min_sales_constr_3: Sold_0[3] = MIN ( Q0_var[3] , 414.605069752064 )
 max_lost_constr_3: Lost_0[3] = MAX ( Lost_0_expr_3 , 0 )
 max_Q1_constr_3: Q1_var[3] = MAX ( Q_hat_adjusted_3 , 0 )
 min_sales2_constr_3: Sold_1[3] = MIN ( total_stock_second_period_3
   , 116.6844884669272 )
 max_lost2_constr_3: Lost_1[3] = MAX ( Lost_1_expr_3 , 0 )
 GC24: Fraction[4] = LOGISTIC ( f_var[4] )
 min_sales_constr_4: Sold_0[4] = MIN ( Q0_var[4] , 1184.825383190634 )
 max_lost_constr_4: Lost_0[4] = MAX ( Lost_0_expr_4 , 0 )
 max_Q1_constr_4: Q1_var[4] = MAX ( Q_hat_adjusted_4 , 0 )
 min_sales2_constr_4: Sold_1[4] = MIN ( total_stock_second_period_4
   , 289.8637644831363 )
 max_lost2_constr_4: Lost_1[4] = MAX ( Lost_1_expr_4 , 0 )
 GC30: Fraction[5] = LOGISTIC ( f_var[5] )
 min_sales_constr_5: Sold_0[5] = MIN ( Q0_var[5] , 530.9830759590976 )
 max_lost_constr_5: Lost_0[5] = MAX ( Lost_0_expr_5 , 0 )
 max_Q1_constr_5: Q1_var[5] = MAX ( Q_hat_adjusted_5 , 0 )
 min_sales2_constr_5: Sold_1[5] = MIN ( total_stock_second_period_5
   , 137.96958279269 )
 max_lost2_constr_5: Lost_1[5] = MAX ( Lost_1_expr_5 , 0 )
 GC36: Fraction[6] = LOGISTIC ( f_var[6] )
 min_sales_constr_6: Sold_0[6] = MIN ( Q0_var[6] , 579.1185032127753 )
 max_lost_constr_6: Lost_0[6] = MAX ( Lost_0_expr_6 , 0 )
 max_Q1_constr_6: Q1_var[6] = MAX ( Q_hat_adjusted_6 , 0 )
 min_sales2_constr_6: Sold_1[6] = MIN ( total_stock_second_period_6
   , 164.0034301584347 )
 max_lost2_constr_6: Lost_1[6] = MAX ( Lost_1_expr_6 , 0 )
 GC42: Fraction[7] = LOGISTIC ( f_var[7] )
 min_sales_constr_7: Sold_0[7] = MIN ( Q0_var[7] , 1179.923877787524 )
 max_lost_constr_7: Lost_0[7] = MAX ( Lost_0_expr_7 , 0 )
 max_Q1_constr_7: Q1_var[7] = MAX ( Q_hat_adjusted_7 , 0 )
 min_sales2_constr_7: Sold_1[7] = MIN ( total_stock_second_period_7
   , 292.3140309082997 )
 max_lost2_constr_7: Lost_1[7] = MAX ( Lost_1_expr_7 , 0 )
 GC48: Fraction[8] = LOGISTIC ( f_var[8] )
 min_sales_constr_8: Sold_0[8] = MIN ( Q0_var[8] , 1177.346133627161 )
 max_lost_constr_8: Lost_0[8] = MAX ( Lost_0_expr_8 , 0 )
 max_Q1_constr_8: Q1_var[8] = MAX ( Q_hat_adjusted_8 , 0 )
 min_sales2_constr_8: Sold_1[8] = MIN ( total_stock_second_period_8
   , 280.7439479491867 )
 max_lost2_constr_8: Lost_1[8] = MAX ( Lost_1_expr_8 , 0 )
 GC54: Fraction[9] = LOGISTIC ( f_var[9] )
 min_sales_constr_9: Sold_0[9] = MIN ( Q0_var[9] , 485.9344047016101 )
 max_lost_constr_9: Lost_0[9] = MAX ( Lost_0_expr_9 , 0 )
 max_Q1_constr_9: Q1_var[9] = MAX ( Q_hat_adjusted_9 , 0 )
 min_sales2_constr_9: Sold_1[9] = MIN ( total_stock_second_period_9
   , 144.863405796243 )
 max_lost2_constr_9: Lost_1[9] = MAX ( Lost_1_expr_9 , 0 )
 GC60: Fraction[10] = LOGISTIC ( f_var[10] )
 min_sales_constr_10: Sold_0[10] = MIN ( Q0_var[10] , 924.8748576553983 )
 max_lost_constr_10: Lost_0[10] = MAX ( Lost_0_expr_10 , 0 )
 max_Q1_constr_10: Q1_var[10] = MAX ( Q_hat_adjusted_10 , 0 )
 min_sales2_constr_10: Sold_1[10] = MIN ( total_stock_second_period_10
   , 249.7327239701096 )
 max_lost2_constr_10: Lost_1[10] = MAX ( Lost_1_expr_10 , 0 )
 GC66: Fraction[11] = LOGISTIC ( f_var[11] )
 min_sales_constr_11: Sold_0[11] = MIN ( Q0_var[11] , 960.7120767671063 )
 max_lost_constr_11: Lost_0[11] = MAX ( Lost_0_expr_11 , 0 )
 max_Q1_constr_11: Q1_var[11] = MAX ( Q_hat_adjusted_11 , 0 )
 min_sales2_constr_11: Sold_1[11] = MIN ( total_stock_second_period_11
   , 264.4235466388491 )
 max_lost2_constr_11: Lost_1[11] = MAX ( Lost_1_expr_11 , 0 )
 GC72: Fraction[12] = LOGISTIC ( f_var[12] )
 min_sales_constr_12: Sold_0[12] = MIN ( Q0_var[12] , 848.9068635451672 )
 max_lost_constr_12: Lost_0[12] = MAX ( Lost_0_expr_12 , 0 )
 max_Q1_constr_12: Q1_var[12] = MAX ( Q_hat_adjusted_12 , 0 )
 min_sales2_constr_12: Sold_1[12] = MIN ( total_stock_second_period_12
   , 226.0895873733634 )
 max_lost2_constr_12: Lost_1[12] = MAX ( Lost_1_expr_12 , 0 )
 GC78: Fraction[13] = LOGISTIC ( f_var[13] )
 min_sales_constr_13: Sold_0[13] = MIN ( Q0_var[13] , 640.6673867048216 )
 max_lost_constr_13: Lost_0[13] = MAX ( Lost_0_expr_13 , 0 )
 max_Q1_constr_13: Q1_var[13] = MAX ( Q_hat_adjusted_13 , 0 )
 min_sales2_constr_13: Sold_1[13] = MIN ( total_stock_second_period_13
   , 152.6644753474297 )
 max_lost2_constr_13: Lost_1[13] = MAX ( Lost_1_expr_13 , 0 )
 GC84: Fraction[14] = LOGISTIC ( f_var[14] )
 min_sales_constr_14: Sold_0[14] = MIN ( Q0_var[14] , 683.4576676850007 )
 max_lost_constr_14: Lost_0[14] = MAX ( Lost_0_expr_14 , 0 )
 max_Q1_constr_14: Q1_var[14] = MAX ( Q_hat_adjusted_14 , 0 )
 min_sales2_constr_14: Sold_1[14] = MIN ( total_stock_second_period_14
   , 190.7393689112113 )
 max_lost2_constr_14: Lost_1[14] = MAX ( Lost_1_expr_14 , 0 )
 GC90: Fraction[15] = LOGISTIC ( f_var[15] )
 min_sales_constr_15: Sold_0[15] = MIN ( Q0_var[15] , 565.2714624479981 )
 max_lost_constr_15: Lost_0[15] = MAX ( Lost_0_expr_15 , 0 )
 max_Q1_constr_15: Q1_var[15] = MAX ( Q_hat_adjusted_15 , 0 )
 min_sales2_constr_15: Sold_1[15] = MIN ( total_stock_second_period_15
   , 150.7883285188988 )
 max_lost2_constr_15: Lost_1[15] = MAX ( Lost_1_expr_15 , 0 )
 GC96: Fraction[16] = LOGISTIC ( f_var[16] )
 min_sales_constr_16: Sold_0[16] = MIN ( Q0_var[16] , 662.207580975467 )
 max_lost_constr_16: Lost_0[16] = MAX ( Lost_0_expr_16 , 0 )
 max_Q1_constr_16: Q1_var[16] = MAX ( Q_hat_adjusted_16 , 0 )
 min_sales2_constr_16: Sold_1[16] = MIN ( total_stock_second_period_16
   , 154.7084069550222 )
 max_lost2_constr_16: Lost_1[16] = MAX ( Lost_1_expr_16 , 0 )
 GC102: Fraction[17] = LOGISTIC ( f_var[17] )
 min_sales_constr_17: Sold_0[17] = MIN ( Q0_var[17] , 558.377137774725 )
 max_lost_constr_17: Lost_0[17] = MAX ( Lost_0_expr_17 , 0 )
 max_Q1_constr_17: Q1_var[17] = MAX ( Q_hat_adjusted_17 , 0 )
 min_sales2_constr_17: Sold_1[17] = MIN ( total_stock_second_period_17
   , 137.1271888611869 )
 max_lost2_constr_17: Lost_1[17] = MAX ( Lost_1_expr_17 , 0 )
 GC108: Fraction[18] = LOGISTIC ( f_var[18] )
 min_sales_constr_18: Sold_0[18] = MIN ( Q0_var[18] , 415.4275787598831 )
 max_lost_constr_18: Lost_0[18] = MAX ( Lost_0_expr_18 , 0 )
 max_Q1_constr_18: Q1_var[18] = MAX ( Q_hat_adjusted_18 , 0 )
 min_sales2_constr_18: Sold_1[18] = MIN ( total_stock_second_period_18
   , 105.1325436115658 )
 max_lost2_constr_18: Lost_1[18] = MAX ( Lost_1_expr_18 , 0 )
 GC114: Fraction[19] = LOGISTIC ( f_var[19] )
 min_sales_constr_19: Sold_0[19] = MIN ( Q0_var[19] , 756.3915232558497 )
 max_lost_constr_19: Lost_0[19] = MAX ( Lost_0_expr_19 , 0 )
 max_Q1_constr_19: Q1_var[19] = MAX ( Q_hat_adjusted_19 , 0 )
 min_sales2_constr_19: Sold_1[19] = MIN ( total_stock_second_period_19
   , 207.3158422217444 )
 max_lost2_constr_19: Lost_1[19] = MAX ( Lost_1_expr_19 , 0 )
 GC120: Fraction[20] = LOGISTIC ( f_var[20] )
 min_sales_constr_20: Sold_0[20] = MIN ( Q0_var[20] , 488.4966268091613 )
 max_lost_constr_20: Lost_0[20] = MAX ( Lost_0_expr_20 , 0 )
 max_Q1_constr_20: Q1_var[20] = MAX ( Q_hat_adjusted_20 , 0 )
 min_sales2_constr_20: Sold_1[20] = MIN ( total_stock_second_period_20
   , 126.4136212633655 )
 max_lost2_constr_20: Lost_1[20] = MAX ( Lost_1_expr_20 , 0 )
 GC126: Fraction[21] = LOGISTIC ( f_var[21] )
 min_sales_constr_21: Sold_0[21] = MIN ( Q0_var[21] , 433.7344164331509 )
 max_lost_constr_21: Lost_0[21] = MAX ( Lost_0_expr_21 , 0 )
 max_Q1_constr_21: Q1_var[21] = MAX ( Q_hat_adjusted_21 , 0 )
 min_sales2_constr_21: Sold_1[21] = MIN ( total_stock_second_period_21
   , 103.285925985933 )
 max_lost2_constr_21: Lost_1[21] = MAX ( Lost_1_expr_21 , 0 )
 GC132: Fraction[22] = LOGISTIC ( f_var[22] )
 min_sales_constr_22: Sold_0[22] = MIN ( Q0_var[22] , 566.1747417586032 )
 max_lost_constr_22: Lost_0[22] = MAX ( Lost_0_expr_22 , 0 )
 max_Q1_constr_22: Q1_var[22] = MAX ( Q_hat_adjusted_22 , 0 )
 min_sales2_constr_22: Sold_1[22] = MIN ( total_stock_second_period_22
   , 135.9207354701485 )
 max_lost2_constr_22: Lost_1[22] = MAX ( Lost_1_expr_22 , 0 )
 GC138: Fraction[23] = LOGISTIC ( f_var[23] )
 min_sales_constr_23: Sold_0[23] = MIN ( Q0_var[23] , 1113.853236533312 )
 max_lost_constr_23: Lost_0[23] = MAX ( Lost_0_expr_23 , 0 )
 max_Q1_constr_23: Q1_var[23] = MAX ( Q_hat_adjusted_23 , 0 )
 min_sales2_constr_23: Sold_1[23] = MIN ( total_stock_second_period_23
   , 275.3010490633181 )
 max_lost2_constr_23: Lost_1[23] = MAX ( Lost_1_expr_23 , 0 )
 GC144: Fraction[24] = LOGISTIC ( f_var[24] )
 min_sales_constr_24: Sold_0[24] = MIN ( Q0_var[24] , 674.7920442704966 )
 max_lost_constr_24: Lost_0[24] = MAX ( Lost_0_expr_24 , 0 )
 max_Q1_constr_24: Q1_var[24] = MAX ( Q_hat_adjusted_24 , 0 )
 min_sales2_constr_24: Sold_1[24] = MIN ( total_stock_second_period_24
   , 179.2119405613942 )
 max_lost2_constr_24: Lost_1[24] = MAX ( Lost_1_expr_24 , 0 )
 GC150: Fraction[25] = LOGISTIC ( f_var[25] )
 min_sales_constr_25: Sold_0[25] = MIN ( Q0_var[25] , 542.0494353456584 )
 max_lost_constr_25: Lost_0[25] = MAX ( Lost_0_expr_25 , 0 )
 max_Q1_constr_25: Q1_var[25] = MAX ( Q_hat_adjusted_25 , 0 )
 min_sales2_constr_25: Sold_1[25] = MIN ( total_stock_second_period_25
   , 128.9695519040114 )
 max_lost2_constr_25: Lost_1[25] = MAX ( Lost_1_expr_25 , 0 )
 GC156: Fraction[26] = LOGISTIC ( f_var[26] )
 min_sales_constr_26: Sold_0[26] = MIN ( Q0_var[26] , 970.052225305534 )
 max_lost_constr_26: Lost_0[26] = MAX ( Lost_0_expr_26 , 0 )
 max_Q1_constr_26: Q1_var[26] = MAX ( Q_hat_adjusted_26 , 0 )
 min_sales2_constr_26: Sold_1[26] = MIN ( total_stock_second_period_26
   , 235.8785546997134 )
 max_lost2_constr_26: Lost_1[26] = MAX ( Lost_1_expr_26 , 0 )
 GC162: Fraction[27] = LOGISTIC ( f_var[27] )
 min_sales_constr_27: Sold_0[27] = MIN ( Q0_var[27] , 1212.708785929688 )
 max_lost_constr_27: Lost_0[27] = MAX ( Lost_0_expr_27 , 0 )
 max_Q1_constr_27: Q1_var[27] = MAX ( Q_hat_adjusted_27 , 0 )
 min_sales2_constr_27: Sold_1[27] = MIN ( total_stock_second_period_27
   , 292.1669316126 )
 max_lost2_constr_27: Lost_1[27] = MAX ( Lost_1_expr_27 , 0 )
 GC168: Fraction[28] = LOGISTIC ( f_var[28] )
 min_sales_constr_28: Sold_0[28] = MIN ( Q0_var[28] , 937.7203361721765 )
 max_lost_constr_28: Lost_0[28] = MAX ( Lost_0_expr_28 , 0 )
 max_Q1_constr_28: Q1_var[28] = MAX ( Q_hat_adjusted_28 , 0 )
 min_sales2_constr_28: Sold_1[28] = MIN ( total_stock_second_period_28
   , 223.5753383834952 )
 max_lost2_constr_28: Lost_1[28] = MAX ( Lost_1_expr_28 , 0 )
 GC174: Fraction[29] = LOGISTIC ( f_var[29] )
 min_sales_constr_29: Sold_0[29] = MIN ( Q0_var[29] , 1102.630414813016 )
 max_lost_constr_29: Lost_0[29] = MAX ( Lost_0_expr_29 , 0 )
 max_Q1_constr_29: Q1_var[29] = MAX ( Q_hat_adjusted_29 , 0 )
 min_sales2_constr_29: Sold_1[29] = MIN ( total_stock_second_period_29
   , 288.2755409412997 )
 max_lost2_constr_29: Lost_1[29] = MAX ( Lost_1_expr_29 , 0 )
 GC180: Fraction[30] = LOGISTIC ( f_var[30] )
 min_sales_constr_30: Sold_0[30] = MIN ( Q0_var[30] , 644.085155158248 )
 max_lost_constr_30: Lost_0[30] = MAX ( Lost_0_expr_30 , 0 )
 max_Q1_constr_30: Q1_var[30] = MAX ( Q_hat_adjusted_30 , 0 )
 min_sales2_constr_30: Sold_1[30] = MIN ( total_stock_second_period_30
   , 148.1657559812276 )
 max_lost2_constr_30: Lost_1[30] = MAX ( Lost_1_expr_30 , 0 )
 GC186: Fraction[31] = LOGISTIC ( f_var[31] )
 min_sales_constr_31: Sold_0[31] = MIN ( Q0_var[31] , 619.6463570481958 )
 max_lost_constr_31: Lost_0[31] = MAX ( Lost_0_expr_31 , 0 )
 max_Q1_constr_31: Q1_var[31] = MAX ( Q_hat_adjusted_31 , 0 )
 min_sales2_constr_31: Sold_1[31] = MIN ( total_stock_second_period_31
   , 158.6040569015613 )
 max_lost2_constr_31: Lost_1[31] = MAX ( Lost_1_expr_31 , 0 )
 GC192: Fraction[32] = LOGISTIC ( f_var[32] )
 min_sales_constr_32: Sold_0[32] = MIN ( Q0_var[32] , 687.618944322122 )
 max_lost_constr_32: Lost_0[32] = MAX ( Lost_0_expr_32 , 0 )
 max_Q1_constr_32: Q1_var[32] = MAX ( Q_hat_adjusted_32 , 0 )
 min_sales2_constr_32: Sold_1[32] = MIN ( total_stock_second_period_32
   , 173.7169212259221 )
 max_lost2_constr_32: Lost_1[32] = MAX ( Lost_1_expr_32 , 0 )
 GC198: Fraction[33] = LOGISTIC ( f_var[33] )
 min_sales_constr_33: Sold_0[33] = MIN ( Q0_var[33] , 404.1361694017194 )
 max_lost_constr_33: Lost_0[33] = MAX ( Lost_0_expr_33 , 0 )
 max_Q1_constr_33: Q1_var[33] = MAX ( Q_hat_adjusted_33 , 0 )
 min_sales2_constr_33: Sold_1[33] = MIN ( total_stock_second_period_33
   , 112.7910533381431 )
 max_lost2_constr_33: Lost_1[33] = MAX ( Lost_1_expr_33 , 0 )
 GC204: Fraction[34] = LOGISTIC ( f_var[34] )
 min_sales_constr_34: Sold_0[34] = MIN ( Q0_var[34] , 1188.151607817763 )
 max_lost_constr_34: Lost_0[34] = MAX ( Lost_0_expr_34 , 0 )
 max_Q1_constr_34: Q1_var[34] = MAX ( Q_hat_adjusted_34 , 0 )
 min_sales2_constr_34: Sold_1[34] = MIN ( total_stock_second_period_34
   , 288.0863890505626 )
 max_lost2_constr_34: Lost_1[34] = MAX ( Lost_1_expr_34 , 0 )
 GC210: Fraction[35] = LOGISTIC ( f_var[35] )
 min_sales_constr_35: Sold_0[35] = MIN ( Q0_var[35] , 519.7215275568076 )
 max_lost_constr_35: Lost_0[35] = MAX ( Lost_0_expr_35 , 0 )
 max_Q1_constr_35: Q1_var[35] = MAX ( Q_hat_adjusted_35 , 0 )
 min_sales2_constr_35: Sold_1[35] = MIN ( total_stock_second_period_35
   , 143.101535422734 )
 max_lost2_constr_35: Lost_1[35] = MAX ( Lost_1_expr_35 , 0 )
 GC216: Fraction[36] = LOGISTIC ( f_var[36] )
 min_sales_constr_36: Sold_0[36] = MIN ( Q0_var[36] , 896.7886466557181 )
 max_lost_constr_36: Lost_0[36] = MAX ( Lost_0_expr_36 , 0 )
 max_Q1_constr_36: Q1_var[36] = MAX ( Q_hat_adjusted_36 , 0 )
 min_sales2_constr_36: Sold_1[36] = MIN ( total_stock_second_period_36
   , 217.6634227108934 )
 max_lost2_constr_36: Lost_1[36] = MAX ( Lost_1_expr_36 , 0 )
 GC222: Fraction[37] = LOGISTIC ( f_var[37] )
 min_sales_constr_37: Sold_0[37] = MIN ( Q0_var[37] , 823.1992212479963 )
 max_lost_constr_37: Lost_0[37] = MAX ( Lost_0_expr_37 , 0 )
 max_Q1_constr_37: Q1_var[37] = MAX ( Q_hat_adjusted_37 , 0 )
 min_sales2_constr_37: Sold_1[37] = MIN ( total_stock_second_period_37
   , 210.4384939845112 )
 max_lost2_constr_37: Lost_1[37] = MAX ( Lost_1_expr_37 , 0 )
 GC228: Fraction[38] = LOGISTIC ( f_var[38] )
 min_sales_constr_38: Sold_0[38] = MIN ( Q0_var[38] , 1115.579999030597 )
 max_lost_constr_38: Lost_0[38] = MAX ( Lost_0_expr_38 , 0 )
 max_Q1_constr_38: Q1_var[38] = MAX ( Q_hat_adjusted_38 , 0 )
 min_sales2_constr_38: Sold_1[38] = MIN ( total_stock_second_period_38
   , 285.9058633584382 )
 max_lost2_constr_38: Lost_1[38] = MAX ( Lost_1_expr_38 , 0 )
 GC234: Fraction[39] = LOGISTIC ( f_var[39] )
 min_sales_constr_39: Sold_0[39] = MIN ( Q0_var[39] , 505.6800124242206 )
 max_lost_constr_39: Lost_0[39] = MAX ( Lost_0_expr_39 , 0 )
 max_Q1_constr_39: Q1_var[39] = MAX ( Q_hat_adjusted_39 , 0 )
 min_sales2_constr_39: Sold_1[39] = MIN ( total_stock_second_period_39
   , 146.3403252941073 )
 max_lost2_constr_39: Lost_1[39] = MAX ( Lost_1_expr_39 , 0 )
 GC240: Fraction[40] = LOGISTIC ( f_var[40] )
 min_sales_constr_40: Sold_0[40] = MIN ( Q0_var[40] , 890.4839503940082 )
 max_lost_constr_40: Lost_0[40] = MAX ( Lost_0_expr_40 , 0 )
 max_Q1_constr_40: Q1_var[40] = MAX ( Q_hat_adjusted_40 , 0 )
 min_sales2_constr_40: Sold_1[40] = MIN ( total_stock_second_period_40
   , 239.2686977630919 )
 max_lost2_constr_40: Lost_1[40] = MAX ( Lost_1_expr_40 , 0 )
 GC246: Fraction[41] = LOGISTIC ( f_var[41] )
 min_sales_constr_41: Sold_0[41] = MIN ( Q0_var[41] , 479.5083968544993 )
 max_lost_constr_41: Lost_0[41] = MAX ( Lost_0_expr_41 , 0 )
 max_Q1_constr_41: Q1_var[41] = MAX ( Q_hat_adjusted_41 , 0 )
 min_sales2_constr_41: Sold_1[41] = MIN ( total_stock_second_period_41
   , 120.0587884463515 )
 max_lost2_constr_41: Lost_1[41] = MAX ( Lost_1_expr_41 , 0 )
 GC252: Fraction[42] = LOGISTIC ( f_var[42] )
 min_sales_constr_42: Sold_0[42] = MIN ( Q0_var[42] , 926.2854918588041 )
 max_lost_constr_42: Lost_0[42] = MAX ( Lost_0_expr_42 , 0 )
 max_Q1_constr_42: Q1_var[42] = MAX ( Q_hat_adjusted_42 , 0 )
 min_sales2_constr_42: Sold_1[42] = MIN ( total_stock_second_period_42
   , 203.475821430823 )
 max_lost2_constr_42: Lost_1[42] = MAX ( Lost_1_expr_42 , 0 )
 GC258: Fraction[43] = LOGISTIC ( f_var[43] )
 min_sales_constr_43: Sold_0[43] = MIN ( Q0_var[43] , 478.4658654171332 )
 max_lost_constr_43: Lost_0[43] = MAX ( Lost_0_expr_43 , 0 )
 max_Q1_constr_43: Q1_var[43] = MAX ( Q_hat_adjusted_43 , 0 )
 min_sales2_constr_43: Sold_1[43] = MIN ( total_stock_second_period_43
   , 123.5063711923435 )
 max_lost2_constr_43: Lost_1[43] = MAX ( Lost_1_expr_43 , 0 )
 GC264: Fraction[44] = LOGISTIC ( f_var[44] )
 min_sales_constr_44: Sold_0[44] = MIN ( Q0_var[44] , 559.628089659188 )
 max_lost_constr_44: Lost_0[44] = MAX ( Lost_0_expr_44 , 0 )
 max_Q1_constr_44: Q1_var[44] = MAX ( Q_hat_adjusted_44 , 0 )
 min_sales2_constr_44: Sold_1[44] = MIN ( total_stock_second_period_44
   , 104.935745678194 )
 max_lost2_constr_44: Lost_1[44] = MAX ( Lost_1_expr_44 , 0 )
 GC270: Fraction[45] = LOGISTIC ( f_var[45] )
 min_sales_constr_45: Sold_0[45] = MIN ( Q0_var[45] , 1010.635998079604 )
 max_lost_constr_45: Lost_0[45] = MAX ( Lost_0_expr_45 , 0 )
 max_Q1_constr_45: Q1_var[45] = MAX ( Q_hat_adjusted_45 , 0 )
 min_sales2_constr_45: Sold_1[45] = MIN ( total_stock_second_period_45
   , 251.6757307672283 )
 max_lost2_constr_45: Lost_1[45] = MAX ( Lost_1_expr_45 , 0 )
 GC276: Fraction[46] = LOGISTIC ( f_var[46] )
 min_sales_constr_46: Sold_0[46] = MIN ( Q0_var[46] , 656.7796503477273 )
 max_lost_constr_46: Lost_0[46] = MAX ( Lost_0_expr_46 , 0 )
 max_Q1_constr_46: Q1_var[46] = MAX ( Q_hat_adjusted_46 , 0 )
 min_sales2_constr_46: Sold_1[46] = MIN ( total_stock_second_period_46
   , 176.6927788344125 )
 max_lost2_constr_46: Lost_1[46] = MAX ( Lost_1_expr_46 , 0 )
 GC282: Fraction[47] = LOGISTIC ( f_var[47] )
 min_sales_constr_47: Sold_0[47] = MIN ( Q0_var[47] , 1080.132772155145 )
 max_lost_constr_47: Lost_0[47] = MAX ( Lost_0_expr_47 , 0 )
 max_Q1_constr_47: Q1_var[47] = MAX ( Q_hat_adjusted_47 , 0 )
 min_sales2_constr_47: Sold_1[47] = MIN ( total_stock_second_period_47
   , 272.7711211846461 )
 max_lost2_constr_47: Lost_1[47] = MAX ( Lost_1_expr_47 , 0 )
 GC288: Fraction[48] = LOGISTIC ( f_var[48] )
 min_sales_constr_48: Sold_0[48] = MIN ( Q0_var[48] , 721.7531147365655 )
 max_lost_constr_48: Lost_0[48] = MAX ( Lost_0_expr_48 , 0 )
 max_Q1_constr_48: Q1_var[48] = MAX ( Q_hat_adjusted_48 , 0 )
 min_sales2_constr_48: Sold_1[48] = MIN ( total_stock_second_period_48
   , 155.525741258146 )
 max_lost2_constr_48: Lost_1[48] = MAX ( Lost_1_expr_48 , 0 )
 GC294: Fraction[49] = LOGISTIC ( f_var[49] )
 min_sales_constr_49: Sold_0[49] = MIN ( Q0_var[49] , 458.6193644477646 )
 max_lost_constr_49: Lost_0[49] = MAX ( Lost_0_expr_49 , 0 )
 max_Q1_constr_49: Q1_var[49] = MAX ( Q_hat_adjusted_49 , 0 )
 min_sales2_constr_49: Sold_1[49] = MIN ( total_stock_second_period_49
   , 113.4499262923876 )
 max_lost2_constr_49: Lost_1[49] = MAX ( Lost_1_expr_49 , 0 )
End
