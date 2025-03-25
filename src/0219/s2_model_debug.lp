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
 R2: 1.547154024076175 alphas[0] + 0.5071945880027401 alphas[1]
   - 0.5279190890110352 alphas[2] + 0.3512344456773365 alphas[3]
   - alphas[4] + f_var[0] = 0
 R3: - 2399.622212800051 Fraction[0] + Q0_var[0] = 0
 R4: Q0_var[0] + Lost_0_expr_0 = 1863.320453002552
 R5: Q0_var[0] + Q_hat_adjusted_0 = 2411.21262694421
 R6: Sold_0[0] - Q0_var[0] - Q1_var[0] + total_stock_second_period_0 = 0
 R7: total_stock_second_period_0 + Lost_1_expr_0 = 416.6595179585461
 R8: Sold_1[0] + Left_1[0] - total_stock_second_period_0 = 0
 R9: - 700 Sold_0[0] - 700 Sold_1[0] + 700 Lost_0[0] + 700 Lost_1[0]
   + 300 Left_1[0] + profits_vars[0] = 0
 Fraction_lower_bound_1: Fraction[1] >= 0
 Fraction_upper_bound_1: Fraction[1] <= 1
 R12: 0.6719167914586636 alphas[0] - 0.7087583091536845 alphas[1]
   - 1.039276067496619 alphas[2] - 0.4666498177755912 alphas[3] - alphas[4]
   + f_var[1] = 0
 R13: - 2399.622212800051 Fraction[1] + Q0_var[1] = 0
 R14: Q0_var[1] + Lost_0_expr_1 = 2419.295741160384
 R15: Q0_var[1] + Q_hat_adjusted_1 = 3057.873343731719
 R16: Sold_0[1] - Q0_var[1] - Q1_var[1] + total_stock_second_period_1 = 0
 R17: total_stock_second_period_1 + Lost_1_expr_1 = 512.0080311282829
 R18: Sold_1[1] + Left_1[1] - total_stock_second_period_1 = 0
 R19: - 700 Sold_0[1] - 700 Sold_1[1] + 700 Lost_0[1] + 700 Lost_1[1]
   + 300 Left_1[1] + profits_vars[1] = 0
 Fraction_lower_bound_2: Fraction[2] >= 0
 Fraction_upper_bound_2: Fraction[2] <= 1
 R22: 1.225789227746987 alphas[0] + 1.33690062399289 alphas[1]
   - 1.118704537264318 alphas[2] - 0.9180035216261917 alphas[3] - alphas[4]
   + f_var[2] = 0
 R23: - 2399.622212800051 Fraction[2] + Q0_var[2] = 0
 R24: Q0_var[2] + Lost_0_expr_2 = 1870.353695908548
 R25: Q0_var[2] + Q_hat_adjusted_2 = 2303.241429659112
 R26: Sold_0[2] - Q0_var[2] - Q1_var[2] + total_stock_second_period_2 = 0
 R27: total_stock_second_period_2 + Lost_1_expr_2 = 562.8635436156746
 R28: Sold_1[2] + Left_1[2] - total_stock_second_period_2 = 0
 R29: - 700 Sold_0[2] - 700 Sold_1[2] + 700 Lost_0[2] + 700 Lost_1[2]
   + 300 Left_1[2] + profits_vars[2] = 0
 Fraction_lower_bound_3: Fraction[3] >= 0
 Fraction_upper_bound_3: Fraction[3] <= 1
 R32: 0.1801481815581999 alphas[0] - 1.331412369201373 alphas[1]
   + 1.571441579717412 alphas[2] + 0.2041289035921307 alphas[3] - alphas[4]
   + f_var[3] = 0
 R33: - 2399.622212800051 Fraction[3] + Q0_var[3] = 0
 R34: Q0_var[3] + Lost_0_expr_3 = 796.4919615763381
 R35: Q0_var[3] + Q_hat_adjusted_3 = 978.6613077796633
 R36: Sold_0[3] - Q0_var[3] - Q1_var[3] + total_stock_second_period_3 = 0
 R37: total_stock_second_period_3 + Lost_1_expr_3 = 0
 R38: Sold_1[3] + Left_1[3] - total_stock_second_period_3 = 0
 R39: - 700 Sold_0[3] - 700 Sold_1[3] + 700 Lost_0[3] + 700 Lost_1[3]
   + 300 Left_1[3] + profits_vars[3] = 0
 Fraction_lower_bound_4: Fraction[4] >= 0
 Fraction_upper_bound_4: Fraction[4] <= 1
 R42: 1.308868035884044 alphas[0] - 0.9884536780602742 alphas[1]
   + 0.8778971908917842 alphas[2] + 1.09348376366179 alphas[3] - alphas[4]
   + f_var[4] = 0
 R43: - 2399.622212800051 Fraction[4] + Q0_var[4] = 0
 R44: Q0_var[4] + Lost_0_expr_4 = 1428.970789513103
 R45: Q0_var[4] + Q_hat_adjusted_4 = 1768.322590555959
 R46: Sold_0[4] - Q0_var[4] - Q1_var[4] + total_stock_second_period_4 = 0
 R47: total_stock_second_period_4 + Lost_1_expr_4 = 240.3308318975413
 R48: Sold_1[4] + Left_1[4] - total_stock_second_period_4 = 0
 R49: - 700 Sold_0[4] - 700 Sold_1[4] + 700 Lost_0[4] + 700 Lost_1[4]
   + 300 Left_1[4] + profits_vars[4] = 0
 Fraction_lower_bound_5: Fraction[5] >= 0
 Fraction_upper_bound_5: Fraction[5] <= 1
 R52: - 0.5558992456041204 alphas[0] - 0.202322233279954 alphas[1]
   + 0.5664057748061654 alphas[2] - 0.2710862317081674 alphas[3]
   - alphas[4] + f_var[5] = 0
 R53: - 2399.622212800051 Fraction[5] + Q0_var[5] = 0
 R54: Q0_var[5] + Lost_0_expr_5 = 1024.947431021874
 R55: Q0_var[5] + Q_hat_adjusted_5 = 1485.351110849815
 R56: Sold_0[5] - Q0_var[5] - Q1_var[5] + total_stock_second_period_5 = 0
 R57: total_stock_second_period_5 + Lost_1_expr_5 = 566.469972703199
 R58: Sold_1[5] + Left_1[5] - total_stock_second_period_5 = 0
 R59: - 700 Sold_0[5] - 700 Sold_1[5] + 700 Lost_0[5] + 700 Lost_1[5]
   + 300 Left_1[5] + profits_vars[5] = 0
 Fraction_lower_bound_6: Fraction[6] >= 0
 Fraction_upper_bound_6: Fraction[6] <= 1
 R62: 0.3169144545798468 alphas[0] + 1.653500030681242 alphas[1]
   + 1.059746793293865 alphas[2] - 0.0608027423517713 alphas[3] - alphas[4]
   + f_var[6] = 0
 R63: - 2399.622212800051 Fraction[6] + Q0_var[6] = 0
 R64: Q0_var[6] + Lost_0_expr_6 = 789.0033504730826
 R65: Q0_var[6] + Q_hat_adjusted_6 = 992.3713302619281
 R66: Sold_0[6] - Q0_var[6] - Q1_var[6] + total_stock_second_period_6 = 0
 R67: total_stock_second_period_6 + Lost_1_expr_6 = 74.21498278084485
 R68: Sold_1[6] + Left_1[6] - total_stock_second_period_6 = 0
 R69: - 700 Sold_0[6] - 700 Sold_1[6] + 700 Lost_0[6] + 700 Lost_1[6]
   + 300 Left_1[6] + profits_vars[6] = 0
 Fraction_lower_bound_7: Fraction[7] >= 0
 Fraction_upper_bound_7: Fraction[7] <= 1
 R72: - 1.704618526090164 alphas[0] - 0.4709637739778613 alphas[1]
   - 0.9555934454577615 alphas[2] - 1.316949368393961 alphas[3] - alphas[4]
   + f_var[7] = 0
 R73: - 2399.622212800051 Fraction[7] + Q0_var[7] = 0
 R74: Q0_var[7] + Lost_0_expr_7 = 1325.236372843044
 R75: Q0_var[7] + Q_hat_adjusted_7 = 1790.691482887521
 R76: Sold_0[7] - Q0_var[7] - Q1_var[7] + total_stock_second_period_7 = 0
 R77: total_stock_second_period_7 + Lost_1_expr_7 = 605.748142250231
 R78: Sold_1[7] + Left_1[7] - total_stock_second_period_7 = 0
 R79: - 700 Sold_0[7] - 700 Sold_1[7] + 700 Lost_0[7] + 700 Lost_1[7]
   + 300 Left_1[7] + profits_vars[7] = 0
 Fraction_lower_bound_8: Fraction[8] >= 0
 Fraction_upper_bound_8: Fraction[8] <= 1
 R82: - 1.990188335913982 alphas[0] - 1.156878261521491 alphas[1]
   - 0.8390859117450984 alphas[2] + 2.053730469627816 alphas[3] - alphas[4]
   + f_var[8] = 0
 R83: - 2399.622212800051 Fraction[8] + Q0_var[8] = 0
 R84: Q0_var[8] + Lost_0_expr_8 = 1948.575588783058
 R85: Q0_var[8] + Q_hat_adjusted_8 = 2363.952241267947
 R86: Sold_0[8] - Q0_var[8] - Q1_var[8] + total_stock_second_period_8 = 0
 R87: total_stock_second_period_8 + Lost_1_expr_8 = 447.7170710950828
 R88: Sold_1[8] + Left_1[8] - total_stock_second_period_8 = 0
 R89: - 700 Sold_0[8] - 700 Sold_1[8] + 700 Lost_0[8] + 700 Lost_1[8]
   + 300 Left_1[8] + profits_vars[8] = 0
 Fraction_lower_bound_9: Fraction[9] >= 0
 Fraction_upper_bound_9: Fraction[9] <= 1
 R92: - 0.3301809370652785 alphas[0] + 0.0982250319123116 alphas[1]
   - 0.0167565184079482 alphas[2] - 1.350961852244915 alphas[3] - alphas[4]
   + f_var[9] = 0
 R93: - 2399.622212800051 Fraction[9] + Q0_var[9] = 0
 R94: Q0_var[9] + Lost_0_expr_9 = 1856.603375943161
 R95: Q0_var[9] + Q_hat_adjusted_9 = 2331.369445415675
 R96: Sold_0[9] - Q0_var[9] - Q1_var[9] + total_stock_second_period_9 = 0
 R97: total_stock_second_period_9 + Lost_1_expr_9 = 484.9084402732783
 R98: Sold_1[9] + Left_1[9] - total_stock_second_period_9 = 0
 R99: - 700 Sold_0[9] - 700 Sold_1[9] + 700 Lost_0[9] + 700 Lost_1[9]
   + 300 Left_1[9] + profits_vars[9] = 0
 Fraction_lower_bound_10: Fraction[10] >= 0
 Fraction_upper_bound_10: Fraction[10] <= 1
 R102: 0.7427896988816222 alphas[0] + 0.6482781932203538 alphas[1]
   - 1.546015466151092 alphas[2] + 0.1403572670839948 alphas[3] - alphas[4]
   + f_var[10] = 0
 R103: - 2399.622212800051 Fraction[10] + Q0_var[10] = 0
 R104: Q0_var[10] + Lost_0_expr_10 = 3049.650376586687
 R105: Q0_var[10] + Q_hat_adjusted_10 = 3625.015831518137
 R106: Sold_0[10] - Q0_var[10] - Q1_var[10] + total_stock_second_period_10
   = 0
 R107: total_stock_second_period_10 + Lost_1_expr_10 = 502.5263370316387
 R108: Sold_1[10] + Left_1[10] - total_stock_second_period_10 = 0
 R109: - 700 Sold_0[10] - 700 Sold_1[10] + 700 Lost_0[10] + 700 Lost_1[10]
   + 300 Left_1[10] + profits_vars[10] = 0
 Fraction_lower_bound_11: Fraction[11] >= 0
 Fraction_upper_bound_11: Fraction[11] <= 1
 R112: - 0.5090171134147718 alphas[0] + 1.403675040078057 alphas[1]
   + 1.206321715533903 alphas[2] + 2.199531954332649 alphas[3] - alphas[4]
   + f_var[11] = 0
 R113: - 2399.622212800051 Fraction[11] + Q0_var[11] = 0
 R114: Q0_var[11] + Lost_0_expr_11 = 1345.712334101624
 R115: Q0_var[11] + Q_hat_adjusted_11 = 1708.972908041338
 R116: Sold_0[11] - Q0_var[11] - Q1_var[11] + total_stock_second_period_11
   = 0
 R117: total_stock_second_period_11 + Lost_1_expr_11 = 274.8327199884636
 R118: Sold_1[11] + Left_1[11] - total_stock_second_period_11 = 0
 R119: - 700 Sold_0[11] - 700 Sold_1[11] + 700 Lost_0[11] + 700 Lost_1[11]
   + 300 Left_1[11] + profits_vars[11] = 0
 Fraction_lower_bound_12: Fraction[12] >= 0
 Fraction_upper_bound_12: Fraction[12] <= 1
 R122: - 1.710042408051459 alphas[0] - 0.645472029462276 alphas[1]
   - 1.272789790770865 alphas[2] + 0.8934800637029057 alphas[3] - alphas[4]
   + f_var[12] = 0
 R123: - 2399.622212800051 Fraction[12] + Q0_var[12] = 0
 R124: Q0_var[12] + Lost_0_expr_12 = 2416.755924917124
 R125: Q0_var[12] + Q_hat_adjusted_12 = 3139.016606285865
 R126: Sold_0[12] - Q0_var[12] - Q1_var[12] + total_stock_second_period_12
   = 0
 R127: total_stock_second_period_12 + Lost_1_expr_12 = 729.2002510259292
 R128: Sold_1[12] + Left_1[12] - total_stock_second_period_12 = 0
 R129: - 700 Sold_0[12] - 700 Sold_1[12] + 700 Lost_0[12] + 700 Lost_1[12]
   + 300 Left_1[12] + profits_vars[12] = 0
 Fraction_lower_bound_13: Fraction[13] >= 0
 Fraction_upper_bound_13: Fraction[13] <= 1
 R132: 0.2225565761853853 alphas[0] - 1.065621044852588 alphas[1]
   + 1.296862229615262 alphas[2] - 0.6134470294472277 alphas[3] - alphas[4]
   + f_var[13] = 0
 R133: - 2399.622212800051 Fraction[13] + Q0_var[13] = 0
 R134: Q0_var[13] + Lost_0_expr_13 = 1007.816163447067
 R135: Q0_var[13] + Q_hat_adjusted_13 = 1430.614794571172
 R136: Sold_0[13] - Q0_var[13] - Q1_var[13] + total_stock_second_period_13
   = 0
 R137: total_stock_second_period_13 + Lost_1_expr_13 = 440.4222413446023
 R138: Sold_1[13] + Left_1[13] - total_stock_second_period_13 = 0
 R139: - 700 Sold_0[13] - 700 Sold_1[13] + 700 Lost_0[13] + 700 Lost_1[13]
   + 300 Left_1[13] + profits_vars[13] = 0
 Fraction_lower_bound_14: Fraction[14] >= 0
 Fraction_upper_bound_14: Fraction[14] <= 1
 R142: - 0.3506628587839953 alphas[0] + 1.468336031437532 alphas[1]
   + 1.143578351342568 alphas[2] + 0.3229476745352464 alphas[3] - alphas[4]
   + f_var[14] = 0
 R143: - 2399.622212800051 Fraction[14] + Q0_var[14] = 0
 R144: Q0_var[14] + Lost_0_expr_14 = 489.2301355085533
 R145: Q0_var[14] + Q_hat_adjusted_14 = 601.0245310768486
 R146: Sold_0[14] - Q0_var[14] - Q1_var[14] + total_stock_second_period_14
   = 0
 R147: total_stock_second_period_14 + Lost_1_expr_14 = 0
 R148: Sold_1[14] + Left_1[14] - total_stock_second_period_14 = 0
 R149: - 700 Sold_0[14] - 700 Sold_1[14] + 700 Lost_0[14] + 700 Lost_1[14]
   + 300 Left_1[14] + profits_vars[14] = 0
 Fraction_lower_bound_15: Fraction[15] >= 0
 Fraction_upper_bound_15: Fraction[15] <= 1
 R152: 0.7552948854746538 alphas[0] - 0.658657150158004 alphas[1]
   - 0.6053249640092098 alphas[2] - 1.628932978073099 alphas[3] - alphas[4]
   + f_var[15] = 0
 R153: - 2399.622212800051 Fraction[15] + Q0_var[15] = 0
 R154: Q0_var[15] + Lost_0_expr_15 = 2410.047733751864
 R155: Q0_var[15] + Q_hat_adjusted_15 = 3001.802869536722
 R156: Sold_0[15] - Q0_var[15] - Q1_var[15] + total_stock_second_period_15
   = 0
 R157: total_stock_second_period_15 + Lost_1_expr_15 = 442.3890127739942
 R158: Sold_1[15] + Left_1[15] - total_stock_second_period_15 = 0
 R159: - 700 Sold_0[15] - 700 Sold_1[15] + 700 Lost_0[15] + 700 Lost_1[15]
   + 300 Left_1[15] + profits_vars[15] = 0
 Fraction_lower_bound_16: Fraction[16] >= 0
 Fraction_upper_bound_16: Fraction[16] <= 1
 R162: 0.0304481953826021 alphas[0] + 1.7369536488944 alphas[1]
   + 1.319480101276632 alphas[2] + 0.5842287150351175 alphas[3] - alphas[4]
   + f_var[16] = 0
 R163: - 2399.622212800051 Fraction[16] + Q0_var[16] = 0
 R164: Q0_var[16] + Lost_0_expr_16 = 1260.37992557425
 R165: Q0_var[16] + Q_hat_adjusted_16 = 1668.414144378171
 R166: Sold_0[16] - Q0_var[16] - Q1_var[16] + total_stock_second_period_16
   = 0
 R167: total_stock_second_period_16 + Lost_1_expr_16 = 129.7839537017187
 R168: Sold_1[16] + Left_1[16] - total_stock_second_period_16 = 0
 R169: - 700 Sold_0[16] - 700 Sold_1[16] + 700 Lost_0[16] + 700 Lost_1[16]
   + 300 Left_1[16] + profits_vars[16] = 0
 Fraction_lower_bound_17: Fraction[17] >= 0
 Fraction_upper_bound_17: Fraction[17] <= 1
 R172: 0.2423539842984611 alphas[0] - 1.330498960541829 alphas[1]
   - 0.1775657417630163 alphas[2] + 0.9651958812573546 alphas[3]
   - alphas[4] + f_var[17] = 0
 R173: - 2399.622212800051 Fraction[17] + Q0_var[17] = 0
 R174: Q0_var[17] + Lost_0_expr_17 = 1999.860327788147
 R175: Q0_var[17] + Q_hat_adjusted_17 = 2528.161330735464
 R176: Sold_0[17] - Q0_var[17] - Q1_var[17] + total_stock_second_period_17
   = 0
 R177: total_stock_second_period_17 + Lost_1_expr_17 = 409.07416498788
 R178: Sold_1[17] + Left_1[17] - total_stock_second_period_17 = 0
 R179: - 700 Sold_0[17] - 700 Sold_1[17] + 700 Lost_0[17] + 700 Lost_1[17]
   + 300 Left_1[17] + profits_vars[17] = 0
 Fraction_lower_bound_18: Fraction[18] >= 0
 Fraction_upper_bound_18: Fraction[18] <= 1
 R182: 0.8469473413459526 alphas[0] - 0.1939304024442014 alphas[1]
   + 0.3001877759706461 alphas[2] - 0.999826642791725 alphas[3] - alphas[4]
   + f_var[18] = 0
 R183: - 2399.622212800051 Fraction[18] + Q0_var[18] = 0
 R184: Q0_var[18] + Lost_0_expr_18 = 1900.301609064119
 R185: Q0_var[18] + Q_hat_adjusted_18 = 2120.724200638424
 R186: Sold_0[18] - Q0_var[18] - Q1_var[18] + total_stock_second_period_18
   = 0
 R187: total_stock_second_period_18 + Lost_1_expr_18 = 278.3027272104797
 R188: Sold_1[18] + Left_1[18] - total_stock_second_period_18 = 0
 R189: - 700 Sold_0[18] - 700 Sold_1[18] + 700 Lost_0[18] + 700 Lost_1[18]
   + 300 Left_1[18] + profits_vars[18] = 0
 Fraction_lower_bound_19: Fraction[19] >= 0
 Fraction_upper_bound_19: Fraction[19] <= 1
 R192: - 0.1684098213932537 alphas[0] + 0.00826099032619569 alphas[1]
   - 0.7111532903639123 alphas[2] + 0.0868702586645638 alphas[3]
   - alphas[4] + f_var[19] = 0
 R193: - 2399.622212800051 Fraction[19] + Q0_var[19] = 0
 R194: Q0_var[19] + Lost_0_expr_19 = 1835.874816900612
 R195: Q0_var[19] + Q_hat_adjusted_19 = 2340.464589697117
 R196: Sold_0[19] - Q0_var[19] - Q1_var[19] + total_stock_second_period_19
   = 0
 R197: total_stock_second_period_19 + Lost_1_expr_19 = 502.278430631151
 R198: Sold_1[19] + Left_1[19] - total_stock_second_period_19 = 0
 R199: - 700 Sold_0[19] - 700 Sold_1[19] + 700 Lost_0[19] + 700 Lost_1[19]
   + 300 Left_1[19] + profits_vars[19] = 0
 Fraction_lower_bound_20: Fraction[20] >= 0
 Fraction_upper_bound_20: Fraction[20] <= 1
 R202: 0.6025570114556836 alphas[0] + 0.836443258982307 alphas[1]
   - 1.276148024692719 alphas[2] - 0.4467024819753025 alphas[3] - alphas[4]
   + f_var[20] = 0
 R203: - 2399.622212800051 Fraction[20] + Q0_var[20] = 0
 R204: Q0_var[20] + Lost_0_expr_20 = 1553.337043545761
 R205: Q0_var[20] + Q_hat_adjusted_20 = 2066.503504929611
 R206: Sold_0[20] - Q0_var[20] - Q1_var[20] + total_stock_second_period_20
   = 0
 R207: total_stock_second_period_20 + Lost_1_expr_20 = 551.6035466487222
 R208: Sold_1[20] + Left_1[20] - total_stock_second_period_20 = 0
 R209: - 700 Sold_0[20] - 700 Sold_1[20] + 700 Lost_0[20] + 700 Lost_1[20]
   + 300 Left_1[20] + profits_vars[20] = 0
 Fraction_lower_bound_21: Fraction[21] >= 0
 Fraction_upper_bound_21: Fraction[21] <= 1
 R212: 0.3554585265442848 alphas[0] - 1.208564594233674 alphas[1]
   - 0.8469792156175543 alphas[2] - 0.9364361582002333 alphas[3]
   - alphas[4] + f_var[21] = 0
 R213: - 2399.622212800051 Fraction[21] + Q0_var[21] = 0
 R214: Q0_var[21] + Lost_0_expr_21 = 2155.482539411831
 R215: Q0_var[21] + Q_hat_adjusted_21 = 2696.073305025719
 R216: Sold_0[21] - Q0_var[21] - Q1_var[21] + total_stock_second_period_21
   = 0
 R217: total_stock_second_period_21 + Lost_1_expr_21 = 390.4334087506201
 R218: Sold_1[21] + Left_1[21] - total_stock_second_period_21 = 0
 R219: - 700 Sold_0[21] - 700 Sold_1[21] + 700 Lost_0[21] + 700 Lost_1[21]
   + 300 Left_1[21] + profits_vars[21] = 0
 Fraction_lower_bound_22: Fraction[22] >= 0
 Fraction_upper_bound_22: Fraction[22] <= 1
 R222: - 0.2437534712457614 alphas[0] + 0.933453489547744 alphas[1]
   - 1.108864352915935 alphas[2] - 0.4034393360234571 alphas[3] - alphas[4]
   + f_var[22] = 0
 R223: - 2399.622212800051 Fraction[22] + Q0_var[22] = 0
 R224: Q0_var[22] + Lost_0_expr_22 = 1448.863627396971
 R225: Q0_var[22] + Q_hat_adjusted_22 = 1826.217830904448
 R226: Sold_0[22] - Q0_var[22] - Q1_var[22] + total_stock_second_period_22
   = 0
 R227: total_stock_second_period_22 + Lost_1_expr_22 = 259.0791454885882
 R228: Sold_1[22] + Left_1[22] - total_stock_second_period_22 = 0
 R229: - 700 Sold_0[22] - 700 Sold_1[22] + 700 Lost_0[22] + 700 Lost_1[22]
   + 300 Left_1[22] + profits_vars[22] = 0
 Fraction_lower_bound_23: Fraction[23] >= 0
 Fraction_upper_bound_23: Fraction[23] <= 1
 R232: - 1.133584261268331 alphas[0] - 0.8819756556195996 alphas[1]
   + 1.57088529290955 alphas[2] + 0.5850328421162463 alphas[3] - alphas[4]
   + f_var[23] = 0
 R233: - 2399.622212800051 Fraction[23] + Q0_var[23] = 0
 R234: Q0_var[23] + Lost_0_expr_23 = 1312.578378716126
 R235: Q0_var[23] + Q_hat_adjusted_23 = 1729.007951973866
 R236: Sold_0[23] - Q0_var[23] - Q1_var[23] + total_stock_second_period_23
   = 0
 R237: total_stock_second_period_23 + Lost_1_expr_23 = 57.68809889334952
 R238: Sold_1[23] + Left_1[23] - total_stock_second_period_23 = 0
 R239: - 700 Sold_0[23] - 700 Sold_1[23] + 700 Lost_0[23] + 700 Lost_1[23]
   + 300 Left_1[23] + profits_vars[23] = 0
 Fraction_lower_bound_24: Fraction[24] >= 0
 Fraction_upper_bound_24: Fraction[24] <= 1
 R242: 0.5113488810628781 alphas[0] - 1.38879304651965 alphas[1]
   + 0.6612240853726261 alphas[2] + 1.76638445673241 alphas[3] - alphas[4]
   + f_var[24] = 0
 R243: - 2399.622212800051 Fraction[24] + Q0_var[24] = 0
 R244: Q0_var[24] + Lost_0_expr_24 = 1320.362128247138
 R245: Q0_var[24] + Q_hat_adjusted_24 = 1402.186042454091
 R246: Sold_0[24] - Q0_var[24] - Q1_var[24] + total_stock_second_period_24
   = 0
 R247: total_stock_second_period_24 + Lost_1_expr_24 = 0
 R248: Sold_1[24] + Left_1[24] - total_stock_second_period_24 = 0
 R249: - 700 Sold_0[24] - 700 Sold_1[24] + 700 Lost_0[24] + 700 Lost_1[24]
   + 300 Left_1[24] + profits_vars[24] = 0
 Fraction_lower_bound_25: Fraction[25] >= 0
 Fraction_upper_bound_25: Fraction[25] <= 1
 R252: - 1.8180946222701 alphas[0] - 0.7374817506009783 alphas[1]
   - 0.0121632511761514 alphas[2] - 0.0415075180097678 alphas[3]
   - alphas[4] + f_var[25] = 0
 R253: - 2399.622212800051 Fraction[25] + Q0_var[25] = 0
 R254: Q0_var[25] + Lost_0_expr_25 = 1919.857222421987
 R255: Q0_var[25] + Q_hat_adjusted_25 = 2498.715484725591
 R256: Sold_0[25] - Q0_var[25] - Q1_var[25] + total_stock_second_period_25
   = 0
 R257: total_stock_second_period_25 + Lost_1_expr_25 = 627.3013999814352
 R258: Sold_1[25] + Left_1[25] - total_stock_second_period_25 = 0
 R259: - 700 Sold_0[25] - 700 Sold_1[25] + 700 Lost_0[25] + 700 Lost_1[25]
   + 300 Left_1[25] + profits_vars[25] = 0
 Fraction_lower_bound_26: Fraction[26] >= 0
 Fraction_upper_bound_26: Fraction[26] <= 1
 R262: - 1.574205718579842 alphas[0] + 0.1162654628382647 alphas[1]
   + 0.3142312424159927 alphas[2] - 1.30055756007045 alphas[3] - alphas[4]
   + f_var[26] = 0
 R263: - 2399.622212800051 Fraction[26] + Q0_var[26] = 0
 R264: Q0_var[26] + Lost_0_expr_26 = 1461.462014480034
 R265: Q0_var[26] + Q_hat_adjusted_26 = 2041.468812950702
 R266: Sold_0[26] - Q0_var[26] - Q1_var[26] + total_stock_second_period_26
   = 0
 R267: total_stock_second_period_26 + Lost_1_expr_26 = 466.9046801371151
 R268: Sold_1[26] + Left_1[26] - total_stock_second_period_26 = 0
 R269: - 700 Sold_0[26] - 700 Sold_1[26] + 700 Lost_0[26] + 700 Lost_1[26]
   + 300 Left_1[26] + profits_vars[26] = 0
 Fraction_lower_bound_27: Fraction[27] >= 0
 Fraction_upper_bound_27: Fraction[27] <= 1
 R272: 1.634626701552279 alphas[0] + 1.578692258154529 alphas[1]
   + 0.9202290947613841 alphas[2] + 0.598985723222408 alphas[3] - alphas[4]
   + f_var[27] = 0
 R273: - 2399.622212800051 Fraction[27] + Q0_var[27] = 0
 R274: Q0_var[27] + Lost_0_expr_27 = 654.9638385887554
 R275: Q0_var[27] + Q_hat_adjusted_27 = 1060.665189490329
 R276: Sold_0[27] - Q0_var[27] - Q1_var[27] + total_stock_second_period_27
   = 0
 R277: total_stock_second_period_27 + Lost_1_expr_27 = 363.318044914416
 R278: Sold_1[27] + Left_1[27] - total_stock_second_period_27 = 0
 R279: - 700 Sold_0[27] - 700 Sold_1[27] + 700 Lost_0[27] + 700 Lost_1[27]
   + 300 Left_1[27] + profits_vars[27] = 0
 Fraction_lower_bound_28: Fraction[28] >= 0
 Fraction_upper_bound_28: Fraction[28] <= 1
 R282: 0.8263246864834869 alphas[0] + 0.2666558971911988 alphas[1]
   + 0.6688406314755746 alphas[2] - 1.353319099926655 alphas[3] - alphas[4]
   + f_var[28] = 0
 R283: - 2399.622212800051 Fraction[28] + Q0_var[28] = 0
 R284: Q0_var[28] + Lost_0_expr_28 = 725.0032584147316
 R285: Q0_var[28] + Q_hat_adjusted_28 = 918.9281131604871
 R286: Sold_0[28] - Q0_var[28] - Q1_var[28] + total_stock_second_period_28
   = 0
 R287: total_stock_second_period_28 + Lost_1_expr_28 = 269.8728228475988
 R288: Sold_1[28] + Left_1[28] - total_stock_second_period_28 = 0
 R289: - 700 Sold_0[28] - 700 Sold_1[28] + 700 Lost_0[28] + 700 Lost_1[28]
   + 300 Left_1[28] + profits_vars[28] = 0
 Fraction_lower_bound_29: Fraction[29] >= 0
 Fraction_upper_bound_29: Fraction[29] <= 1
 R292: 0.0671601157099123 alphas[0] + 0.3769487143676545 alphas[1]
   - 1.42299219254013 alphas[2] + 0.2630299193765472 alphas[3] - alphas[4]
   + f_var[29] = 0
 R293: - 2399.622212800051 Fraction[29] + Q0_var[29] = 0
 R294: Q0_var[29] + Lost_0_expr_29 = 2687.862684089438
 R295: Q0_var[29] + Q_hat_adjusted_29 = 3564.546046014088
 R296: Sold_0[29] - Q0_var[29] - Q1_var[29] + total_stock_second_period_29
   = 0
 R297: total_stock_second_period_29 + Lost_1_expr_29 = 749.3407664332572
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
 Q0_var[0] <= 2400.622212800051
 Q0_var[1] <= 2400.622212800051
 Q0_var[2] <= 2400.622212800051
 Q0_var[3] <= 2400.622212800051
 Q0_var[4] <= 2400.622212800051
 Q0_var[5] <= 2400.622212800051
 Q0_var[6] <= 2400.622212800051
 Q0_var[7] <= 2400.622212800051
 Q0_var[8] <= 2400.622212800051
 Q0_var[9] <= 2400.622212800051
 Q0_var[10] <= 2400.622212800051
 Q0_var[11] <= 2400.622212800051
 Q0_var[12] <= 2400.622212800051
 Q0_var[13] <= 2400.622212800051
 Q0_var[14] <= 2400.622212800051
 Q0_var[15] <= 2400.622212800051
 Q0_var[16] <= 2400.622212800051
 Q0_var[17] <= 2400.622212800051
 Q0_var[18] <= 2400.622212800051
 Q0_var[19] <= 2400.622212800051
 Q0_var[20] <= 2400.622212800051
 Q0_var[21] <= 2400.622212800051
 Q0_var[22] <= 2400.622212800051
 Q0_var[23] <= 2400.622212800051
 Q0_var[24] <= 2400.622212800051
 Q0_var[25] <= 2400.622212800051
 Q0_var[26] <= 2400.622212800051
 Q0_var[27] <= 2400.622212800051
 Q0_var[28] <= 2400.622212800051
 Q0_var[29] <= 2400.622212800051
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
 min_sales_constr_0: Sold_0[0] = MIN ( Q0_var[0] , 1863.320453002552 )
 max_lost_constr_0: Lost_0[0] = MAX ( Lost_0_expr_0 , 0 )
 max_Q1_constr_0: Q1_var[0] = MAX ( Q_hat_adjusted_0 , 0 )
 min_sales2_constr_0: Sold_1[0] = MIN ( total_stock_second_period_0
   , 416.6595179585461 )
 max_lost2_constr_0: Lost_1[0] = MAX ( Lost_1_expr_0 , 0 )
 GC6: Fraction[1] = LOGISTIC ( f_var[1] )
 min_sales_constr_1: Sold_0[1] = MIN ( Q0_var[1] , 2419.295741160384 )
 max_lost_constr_1: Lost_0[1] = MAX ( Lost_0_expr_1 , 0 )
 max_Q1_constr_1: Q1_var[1] = MAX ( Q_hat_adjusted_1 , 0 )
 min_sales2_constr_1: Sold_1[1] = MIN ( total_stock_second_period_1
   , 512.0080311282829 )
 max_lost2_constr_1: Lost_1[1] = MAX ( Lost_1_expr_1 , 0 )
 GC12: Fraction[2] = LOGISTIC ( f_var[2] )
 min_sales_constr_2: Sold_0[2] = MIN ( Q0_var[2] , 1870.353695908548 )
 max_lost_constr_2: Lost_0[2] = MAX ( Lost_0_expr_2 , 0 )
 max_Q1_constr_2: Q1_var[2] = MAX ( Q_hat_adjusted_2 , 0 )
 min_sales2_constr_2: Sold_1[2] = MIN ( total_stock_second_period_2
   , 562.8635436156746 )
 max_lost2_constr_2: Lost_1[2] = MAX ( Lost_1_expr_2 , 0 )
 GC18: Fraction[3] = LOGISTIC ( f_var[3] )
 min_sales_constr_3: Sold_0[3] = MIN ( Q0_var[3] , 796.4919615763381 )
 max_lost_constr_3: Lost_0[3] = MAX ( Lost_0_expr_3 , 0 )
 max_Q1_constr_3: Q1_var[3] = MAX ( Q_hat_adjusted_3 , 0 )
 min_sales2_constr_3: Sold_1[3] = MIN ( total_stock_second_period_3 , 0 )
 max_lost2_constr_3: Lost_1[3] = MAX ( Lost_1_expr_3 , 0 )
 GC24: Fraction[4] = LOGISTIC ( f_var[4] )
 min_sales_constr_4: Sold_0[4] = MIN ( Q0_var[4] , 1428.970789513103 )
 max_lost_constr_4: Lost_0[4] = MAX ( Lost_0_expr_4 , 0 )
 max_Q1_constr_4: Q1_var[4] = MAX ( Q_hat_adjusted_4 , 0 )
 min_sales2_constr_4: Sold_1[4] = MIN ( total_stock_second_period_4
   , 240.3308318975413 )
 max_lost2_constr_4: Lost_1[4] = MAX ( Lost_1_expr_4 , 0 )
 GC30: Fraction[5] = LOGISTIC ( f_var[5] )
 min_sales_constr_5: Sold_0[5] = MIN ( Q0_var[5] , 1024.947431021874 )
 max_lost_constr_5: Lost_0[5] = MAX ( Lost_0_expr_5 , 0 )
 max_Q1_constr_5: Q1_var[5] = MAX ( Q_hat_adjusted_5 , 0 )
 min_sales2_constr_5: Sold_1[5] = MIN ( total_stock_second_period_5
   , 566.469972703199 )
 max_lost2_constr_5: Lost_1[5] = MAX ( Lost_1_expr_5 , 0 )
 GC36: Fraction[6] = LOGISTIC ( f_var[6] )
 min_sales_constr_6: Sold_0[6] = MIN ( Q0_var[6] , 789.0033504730826 )
 max_lost_constr_6: Lost_0[6] = MAX ( Lost_0_expr_6 , 0 )
 max_Q1_constr_6: Q1_var[6] = MAX ( Q_hat_adjusted_6 , 0 )
 min_sales2_constr_6: Sold_1[6] = MIN ( total_stock_second_period_6
   , 74.21498278084485 )
 max_lost2_constr_6: Lost_1[6] = MAX ( Lost_1_expr_6 , 0 )
 GC42: Fraction[7] = LOGISTIC ( f_var[7] )
 min_sales_constr_7: Sold_0[7] = MIN ( Q0_var[7] , 1325.236372843044 )
 max_lost_constr_7: Lost_0[7] = MAX ( Lost_0_expr_7 , 0 )
 max_Q1_constr_7: Q1_var[7] = MAX ( Q_hat_adjusted_7 , 0 )
 min_sales2_constr_7: Sold_1[7] = MIN ( total_stock_second_period_7
   , 605.748142250231 )
 max_lost2_constr_7: Lost_1[7] = MAX ( Lost_1_expr_7 , 0 )
 GC48: Fraction[8] = LOGISTIC ( f_var[8] )
 min_sales_constr_8: Sold_0[8] = MIN ( Q0_var[8] , 1948.575588783058 )
 max_lost_constr_8: Lost_0[8] = MAX ( Lost_0_expr_8 , 0 )
 max_Q1_constr_8: Q1_var[8] = MAX ( Q_hat_adjusted_8 , 0 )
 min_sales2_constr_8: Sold_1[8] = MIN ( total_stock_second_period_8
   , 447.7170710950828 )
 max_lost2_constr_8: Lost_1[8] = MAX ( Lost_1_expr_8 , 0 )
 GC54: Fraction[9] = LOGISTIC ( f_var[9] )
 min_sales_constr_9: Sold_0[9] = MIN ( Q0_var[9] , 1856.603375943161 )
 max_lost_constr_9: Lost_0[9] = MAX ( Lost_0_expr_9 , 0 )
 max_Q1_constr_9: Q1_var[9] = MAX ( Q_hat_adjusted_9 , 0 )
 min_sales2_constr_9: Sold_1[9] = MIN ( total_stock_second_period_9
   , 484.9084402732783 )
 max_lost2_constr_9: Lost_1[9] = MAX ( Lost_1_expr_9 , 0 )
 GC60: Fraction[10] = LOGISTIC ( f_var[10] )
 min_sales_constr_10: Sold_0[10] = MIN ( Q0_var[10] , 3049.650376586687 )
 max_lost_constr_10: Lost_0[10] = MAX ( Lost_0_expr_10 , 0 )
 max_Q1_constr_10: Q1_var[10] = MAX ( Q_hat_adjusted_10 , 0 )
 min_sales2_constr_10: Sold_1[10] = MIN ( total_stock_second_period_10
   , 502.5263370316387 )
 max_lost2_constr_10: Lost_1[10] = MAX ( Lost_1_expr_10 , 0 )
 GC66: Fraction[11] = LOGISTIC ( f_var[11] )
 min_sales_constr_11: Sold_0[11] = MIN ( Q0_var[11] , 1345.712334101624 )
 max_lost_constr_11: Lost_0[11] = MAX ( Lost_0_expr_11 , 0 )
 max_Q1_constr_11: Q1_var[11] = MAX ( Q_hat_adjusted_11 , 0 )
 min_sales2_constr_11: Sold_1[11] = MIN ( total_stock_second_period_11
   , 274.8327199884636 )
 max_lost2_constr_11: Lost_1[11] = MAX ( Lost_1_expr_11 , 0 )
 GC72: Fraction[12] = LOGISTIC ( f_var[12] )
 min_sales_constr_12: Sold_0[12] = MIN ( Q0_var[12] , 2416.755924917124 )
 max_lost_constr_12: Lost_0[12] = MAX ( Lost_0_expr_12 , 0 )
 max_Q1_constr_12: Q1_var[12] = MAX ( Q_hat_adjusted_12 , 0 )
 min_sales2_constr_12: Sold_1[12] = MIN ( total_stock_second_period_12
   , 729.2002510259292 )
 max_lost2_constr_12: Lost_1[12] = MAX ( Lost_1_expr_12 , 0 )
 GC78: Fraction[13] = LOGISTIC ( f_var[13] )
 min_sales_constr_13: Sold_0[13] = MIN ( Q0_var[13] , 1007.816163447067 )
 max_lost_constr_13: Lost_0[13] = MAX ( Lost_0_expr_13 , 0 )
 max_Q1_constr_13: Q1_var[13] = MAX ( Q_hat_adjusted_13 , 0 )
 min_sales2_constr_13: Sold_1[13] = MIN ( total_stock_second_period_13
   , 440.4222413446023 )
 max_lost2_constr_13: Lost_1[13] = MAX ( Lost_1_expr_13 , 0 )
 GC84: Fraction[14] = LOGISTIC ( f_var[14] )
 min_sales_constr_14: Sold_0[14] = MIN ( Q0_var[14] , 489.2301355085533 )
 max_lost_constr_14: Lost_0[14] = MAX ( Lost_0_expr_14 , 0 )
 max_Q1_constr_14: Q1_var[14] = MAX ( Q_hat_adjusted_14 , 0 )
 min_sales2_constr_14: Sold_1[14] = MIN ( total_stock_second_period_14 , 0 )
 max_lost2_constr_14: Lost_1[14] = MAX ( Lost_1_expr_14 , 0 )
 GC90: Fraction[15] = LOGISTIC ( f_var[15] )
 min_sales_constr_15: Sold_0[15] = MIN ( Q0_var[15] , 2410.047733751864 )
 max_lost_constr_15: Lost_0[15] = MAX ( Lost_0_expr_15 , 0 )
 max_Q1_constr_15: Q1_var[15] = MAX ( Q_hat_adjusted_15 , 0 )
 min_sales2_constr_15: Sold_1[15] = MIN ( total_stock_second_period_15
   , 442.3890127739942 )
 max_lost2_constr_15: Lost_1[15] = MAX ( Lost_1_expr_15 , 0 )
 GC96: Fraction[16] = LOGISTIC ( f_var[16] )
 min_sales_constr_16: Sold_0[16] = MIN ( Q0_var[16] , 1260.37992557425 )
 max_lost_constr_16: Lost_0[16] = MAX ( Lost_0_expr_16 , 0 )
 max_Q1_constr_16: Q1_var[16] = MAX ( Q_hat_adjusted_16 , 0 )
 min_sales2_constr_16: Sold_1[16] = MIN ( total_stock_second_period_16
   , 129.7839537017187 )
 max_lost2_constr_16: Lost_1[16] = MAX ( Lost_1_expr_16 , 0 )
 GC102: Fraction[17] = LOGISTIC ( f_var[17] )
 min_sales_constr_17: Sold_0[17] = MIN ( Q0_var[17] , 1999.860327788147 )
 max_lost_constr_17: Lost_0[17] = MAX ( Lost_0_expr_17 , 0 )
 max_Q1_constr_17: Q1_var[17] = MAX ( Q_hat_adjusted_17 , 0 )
 min_sales2_constr_17: Sold_1[17] = MIN ( total_stock_second_period_17
   , 409.07416498788 )
 max_lost2_constr_17: Lost_1[17] = MAX ( Lost_1_expr_17 , 0 )
 GC108: Fraction[18] = LOGISTIC ( f_var[18] )
 min_sales_constr_18: Sold_0[18] = MIN ( Q0_var[18] , 1900.301609064119 )
 max_lost_constr_18: Lost_0[18] = MAX ( Lost_0_expr_18 , 0 )
 max_Q1_constr_18: Q1_var[18] = MAX ( Q_hat_adjusted_18 , 0 )
 min_sales2_constr_18: Sold_1[18] = MIN ( total_stock_second_period_18
   , 278.3027272104797 )
 max_lost2_constr_18: Lost_1[18] = MAX ( Lost_1_expr_18 , 0 )
 GC114: Fraction[19] = LOGISTIC ( f_var[19] )
 min_sales_constr_19: Sold_0[19] = MIN ( Q0_var[19] , 1835.874816900612 )
 max_lost_constr_19: Lost_0[19] = MAX ( Lost_0_expr_19 , 0 )
 max_Q1_constr_19: Q1_var[19] = MAX ( Q_hat_adjusted_19 , 0 )
 min_sales2_constr_19: Sold_1[19] = MIN ( total_stock_second_period_19
   , 502.278430631151 )
 max_lost2_constr_19: Lost_1[19] = MAX ( Lost_1_expr_19 , 0 )
 GC120: Fraction[20] = LOGISTIC ( f_var[20] )
 min_sales_constr_20: Sold_0[20] = MIN ( Q0_var[20] , 1553.337043545761 )
 max_lost_constr_20: Lost_0[20] = MAX ( Lost_0_expr_20 , 0 )
 max_Q1_constr_20: Q1_var[20] = MAX ( Q_hat_adjusted_20 , 0 )
 min_sales2_constr_20: Sold_1[20] = MIN ( total_stock_second_period_20
   , 551.6035466487222 )
 max_lost2_constr_20: Lost_1[20] = MAX ( Lost_1_expr_20 , 0 )
 GC126: Fraction[21] = LOGISTIC ( f_var[21] )
 min_sales_constr_21: Sold_0[21] = MIN ( Q0_var[21] , 2155.482539411831 )
 max_lost_constr_21: Lost_0[21] = MAX ( Lost_0_expr_21 , 0 )
 max_Q1_constr_21: Q1_var[21] = MAX ( Q_hat_adjusted_21 , 0 )
 min_sales2_constr_21: Sold_1[21] = MIN ( total_stock_second_period_21
   , 390.4334087506201 )
 max_lost2_constr_21: Lost_1[21] = MAX ( Lost_1_expr_21 , 0 )
 GC132: Fraction[22] = LOGISTIC ( f_var[22] )
 min_sales_constr_22: Sold_0[22] = MIN ( Q0_var[22] , 1448.863627396971 )
 max_lost_constr_22: Lost_0[22] = MAX ( Lost_0_expr_22 , 0 )
 max_Q1_constr_22: Q1_var[22] = MAX ( Q_hat_adjusted_22 , 0 )
 min_sales2_constr_22: Sold_1[22] = MIN ( total_stock_second_period_22
   , 259.0791454885882 )
 max_lost2_constr_22: Lost_1[22] = MAX ( Lost_1_expr_22 , 0 )
 GC138: Fraction[23] = LOGISTIC ( f_var[23] )
 min_sales_constr_23: Sold_0[23] = MIN ( Q0_var[23] , 1312.578378716126 )
 max_lost_constr_23: Lost_0[23] = MAX ( Lost_0_expr_23 , 0 )
 max_Q1_constr_23: Q1_var[23] = MAX ( Q_hat_adjusted_23 , 0 )
 min_sales2_constr_23: Sold_1[23] = MIN ( total_stock_second_period_23
   , 57.68809889334952 )
 max_lost2_constr_23: Lost_1[23] = MAX ( Lost_1_expr_23 , 0 )
 GC144: Fraction[24] = LOGISTIC ( f_var[24] )
 min_sales_constr_24: Sold_0[24] = MIN ( Q0_var[24] , 1320.362128247138 )
 max_lost_constr_24: Lost_0[24] = MAX ( Lost_0_expr_24 , 0 )
 max_Q1_constr_24: Q1_var[24] = MAX ( Q_hat_adjusted_24 , 0 )
 min_sales2_constr_24: Sold_1[24] = MIN ( total_stock_second_period_24 , 0 )
 max_lost2_constr_24: Lost_1[24] = MAX ( Lost_1_expr_24 , 0 )
 GC150: Fraction[25] = LOGISTIC ( f_var[25] )
 min_sales_constr_25: Sold_0[25] = MIN ( Q0_var[25] , 1919.857222421987 )
 max_lost_constr_25: Lost_0[25] = MAX ( Lost_0_expr_25 , 0 )
 max_Q1_constr_25: Q1_var[25] = MAX ( Q_hat_adjusted_25 , 0 )
 min_sales2_constr_25: Sold_1[25] = MIN ( total_stock_second_period_25
   , 627.3013999814352 )
 max_lost2_constr_25: Lost_1[25] = MAX ( Lost_1_expr_25 , 0 )
 GC156: Fraction[26] = LOGISTIC ( f_var[26] )
 min_sales_constr_26: Sold_0[26] = MIN ( Q0_var[26] , 1461.462014480034 )
 max_lost_constr_26: Lost_0[26] = MAX ( Lost_0_expr_26 , 0 )
 max_Q1_constr_26: Q1_var[26] = MAX ( Q_hat_adjusted_26 , 0 )
 min_sales2_constr_26: Sold_1[26] = MIN ( total_stock_second_period_26
   , 466.9046801371151 )
 max_lost2_constr_26: Lost_1[26] = MAX ( Lost_1_expr_26 , 0 )
 GC162: Fraction[27] = LOGISTIC ( f_var[27] )
 min_sales_constr_27: Sold_0[27] = MIN ( Q0_var[27] , 654.9638385887554 )
 max_lost_constr_27: Lost_0[27] = MAX ( Lost_0_expr_27 , 0 )
 max_Q1_constr_27: Q1_var[27] = MAX ( Q_hat_adjusted_27 , 0 )
 min_sales2_constr_27: Sold_1[27] = MIN ( total_stock_second_period_27
   , 363.318044914416 )
 max_lost2_constr_27: Lost_1[27] = MAX ( Lost_1_expr_27 , 0 )
 GC168: Fraction[28] = LOGISTIC ( f_var[28] )
 min_sales_constr_28: Sold_0[28] = MIN ( Q0_var[28] , 725.0032584147316 )
 max_lost_constr_28: Lost_0[28] = MAX ( Lost_0_expr_28 , 0 )
 max_Q1_constr_28: Q1_var[28] = MAX ( Q_hat_adjusted_28 , 0 )
 min_sales2_constr_28: Sold_1[28] = MIN ( total_stock_second_period_28
   , 269.8728228475988 )
 max_lost2_constr_28: Lost_1[28] = MAX ( Lost_1_expr_28 , 0 )
 GC174: Fraction[29] = LOGISTIC ( f_var[29] )
 min_sales_constr_29: Sold_0[29] = MIN ( Q0_var[29] , 2687.862684089438 )
 max_lost_constr_29: Lost_0[29] = MAX ( Lost_0_expr_29 , 0 )
 max_Q1_constr_29: Q1_var[29] = MAX ( Q_hat_adjusted_29 , 0 )
 min_sales2_constr_29: Sold_1[29] = MIN ( total_stock_second_period_29
   , 749.3407664332572 )
 max_lost2_constr_29: Lost_1[29] = MAX ( Lost_1_expr_29 , 0 )
End
