\ Model profit_maximization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - abs_alpha[0] - abs_alpha[1] - abs_alpha[2] - abs_alpha[3] - abs_alpha[4]
   + profits_vars[0] + profits_vars[1] + profits_vars[2] + profits_vars[3]
   + profits_vars[4] + profits_vars[5] + profits_vars[6] + profits_vars[7]
   + profits_vars[8] + profits_vars[9] + profits_vars[10]
   + profits_vars[11] + profits_vars[12] + profits_vars[13]
   + profits_vars[14] + profits_vars[15] + profits_vars[16]
   + profits_vars[17] + profits_vars[18] + profits_vars[19]
   + profits_vars[20] + profits_vars[21] + profits_vars[22]
   + profits_vars[23] + profits_vars[24]
Subject To
 R0: - alphas[0] + abs_alpha[0] >= 0
 R1: alphas[0] + abs_alpha[0] >= 0
 R2: - alphas[1] + abs_alpha[1] >= 0
 R3: alphas[1] + abs_alpha[1] >= 0
 R4: - alphas[2] + abs_alpha[2] >= 0
 R5: alphas[2] + abs_alpha[2] >= 0
 R6: - alphas[3] + abs_alpha[3] >= 0
 R7: alphas[3] + abs_alpha[3] >= 0
 R8: - alphas[4] + abs_alpha[4] >= 0
 R9: alphas[4] + abs_alpha[4] >= 0
 Fraction_lower_bound_0: Fraction[0] >= 0
 Fraction_upper_bound_0: Fraction[0] <= 1
 R12: - 13.19960510663762 alphas[0] + 29.21624415491359 alphas[1]
   - 125.2791656971945 alphas[2] - 7.239246880603801 alphas[3] - alphas[4]
   + f_var[0] = 0
 R13: - 3725.621915456575 Fraction[0] + Q0_var[0] = 0
 Constr_Left_0_diff_aux_0: Sold_0[0] - Q0_var[0] + Left_0_aux_0 = 0
 Constr_Lost_0_diff_aux_0: Q0_var[0] + Lost_0_aux_0 = 1001.760973706182
 R16: Q0_var[0] + Q_hat_adjusted_0 = 1252.321798038432
 Constr_Q1_plus_left_0: - Left_0[0] - Q1_var[0] + Q1_plus_left[0] = 0
 Constr_Left_1_diff_aux_0: Sold_1[0] - Q1_plus_left[0] + Left_0_aux_0 = 0
 Constr_Lost_1_diff_aux_0: Q1_plus_left[0] + Lost_0_aux_0
   = 250.5583313943889
 Profit_Constraint_0: - 800 Sold_0[0] + 800 Lost_0[0] - 800 Sold_1[0]
   + 200 Left_1[0] + 800 Lost_1[0] + profits_vars[0] = 0
 R21: - 800 Sold_0[0] + 800 Lost_0[0] - 800 Sold_1[0] + 200 Left_1[0]
   + 800 Lost_1[0] + profits_vars[0] = 0
 Fraction_lower_bound_1: Fraction[1] >= 0
 Fraction_upper_bound_1: Fraction[1] <= 1
 R24: - 10.71676643704523 alphas[0] + 24.34386074847181 alphas[1]
   - 98.74882763878153 alphas[2] - 8.453516806969027 alphas[3] - alphas[4]
   + f_var[1] = 0
 R25: - 3725.621915456575 Fraction[1] + Q0_var[1] = 0
 Constr_Left_0_diff_aux_1: Sold_0[1] - Q0_var[1] + Left_0_aux_1 = 0
 Constr_Lost_0_diff_aux_1: Q0_var[1] + Lost_0_aux_1 = 794.8359660065993
 R28: Q0_var[1] + Q_hat_adjusted_1 = 992.3361142244133
 Constr_Q1_plus_left_1: - Left_0[1] - Q1_var[1] + Q1_plus_left[1] = 0
 Constr_Left_1_diff_aux_1: Sold_1[1] - Q1_plus_left[1] + Left_0_aux_1 = 0
 Constr_Lost_1_diff_aux_1: Q1_plus_left[1] + Lost_0_aux_1
   = 197.4976552775631
 Profit_Constraint_1: - 800 Sold_0[1] + 800 Lost_0[1] - 800 Sold_1[1]
   + 200 Left_1[1] + 800 Lost_1[1] + profits_vars[1] = 0
 R33: - 800 Sold_0[1] + 800 Lost_0[1] - 800 Sold_1[1] + 200 Left_1[1]
   + 800 Lost_1[1] + profits_vars[1] = 0
 Fraction_lower_bound_2: Fraction[2] >= 0
 Fraction_upper_bound_2: Fraction[2] <= 1
 R36: - 14.72334458524792 alphas[0] + 27.59606877573662 alphas[1]
   - 51.55076771135585 alphas[2] - 14.28081293465591 alphas[3] - alphas[4]
   + f_var[2] = 0
 R37: - 3725.621915456575 Fraction[2] + Q0_var[2] = 0
 Constr_Left_0_diff_aux_2: Sold_0[2] - Q0_var[2] + Left_0_aux_2 = 0
 Constr_Lost_0_diff_aux_2: Q0_var[2] + Lost_0_aux_2 = 407.1404742176073
 R40: Q0_var[2] + Q_hat_adjusted_2 = 510.2445027195425
 Constr_Q1_plus_left_2: - Left_0[2] - Q1_var[2] + Q1_plus_left[2] = 0
 Constr_Left_1_diff_aux_2: Sold_1[2] - Q1_plus_left[2] + Left_0_aux_2 = 0
 Constr_Lost_1_diff_aux_2: Q1_plus_left[2] + Lost_0_aux_2
   = 103.1015354227212
 Profit_Constraint_2: - 800 Sold_0[2] + 800 Lost_0[2] - 800 Sold_1[2]
   + 200 Left_1[2] + 800 Lost_1[2] + profits_vars[2] = 0
 R45: - 800 Sold_0[2] + 800 Lost_0[2] - 800 Sold_1[2] + 200 Left_1[2]
   + 800 Lost_1[2] + profits_vars[2] = 0
 Fraction_lower_bound_3: Fraction[3] >= 0
 Fraction_upper_bound_3: Fraction[3] <= 1
 R48: - 12.60924160875036 alphas[0] + 24.0771960495571 alphas[1]
   - 124.7370590488924 alphas[2] - 12.04414401923533 alphas[3] - alphas[4]
   + f_var[3] = 0
 R49: - 3725.621915456575 Fraction[3] + Q0_var[3] = 0
 Constr_Left_0_diff_aux_3: Sold_0[3] - Q0_var[3] + Left_0_aux_3 = 0
 Constr_Lost_0_diff_aux_3: Q0_var[3] + Lost_0_aux_3 = 1001.660824741478
 R52: Q0_var[3] + Q_hat_adjusted_3 = 1251.137435917504
 Constr_Q1_plus_left_3: - Left_0[3] - Q1_var[3] + Q1_plus_left[3] = 0
 Constr_Left_1_diff_aux_3: Sold_1[3] - Q1_plus_left[3] + Left_0_aux_3 = 0
 Constr_Lost_1_diff_aux_3: Q1_plus_left[3] + Lost_0_aux_3
   = 249.4741180977848
 Profit_Constraint_3: - 800 Sold_0[3] + 800 Lost_0[3] - 800 Sold_1[3]
   + 200 Left_1[3] + 800 Lost_1[3] + profits_vars[3] = 0
 R57: - 800 Sold_0[3] + 800 Lost_0[3] - 800 Sold_1[3] + 200 Left_1[3]
   + 800 Lost_1[3] + profits_vars[3] = 0
 Fraction_lower_bound_4: Fraction[4] >= 0
 Fraction_upper_bound_4: Fraction[4] <= 1
 R60: - 12.07330969995262 alphas[0] + 27.62544562091512 alphas[1]
   - 103.0855806770158 alphas[2] - 5.318389295313079 alphas[3] - alphas[4]
   + f_var[4] = 0
 R61: - 3725.621915456575 Fraction[4] + Q0_var[4] = 0
 Constr_Left_0_diff_aux_4: Sold_0[4] - Q0_var[4] + Left_0_aux_4 = 0
 Constr_Lost_0_diff_aux_4: Q0_var[4] + Lost_0_aux_4 = 811.2887843628971
 R64: Q0_var[4] + Q_hat_adjusted_4 = 1017.462438569441
 Constr_Q1_plus_left_4: - Left_0[4] - Q1_var[4] + Q1_plus_left[4] = 0
 Constr_Left_1_diff_aux_4: Sold_1[4] - Q1_plus_left[4] + Left_0_aux_4 = 0
 Constr_Lost_1_diff_aux_4: Q1_plus_left[4] + Lost_0_aux_4
   = 206.1711613540316
 Profit_Constraint_4: - 800 Sold_0[4] + 800 Lost_0[4] - 800 Sold_1[4]
   + 200 Left_1[4] + 800 Lost_1[4] + profits_vars[4] = 0
 R69: - 800 Sold_0[4] + 800 Lost_0[4] - 800 Sold_1[4] + 200 Left_1[4]
   + 800 Lost_1[4] + profits_vars[4] = 0
 Fraction_lower_bound_5: Fraction[5] >= 0
 Fraction_upper_bound_5: Fraction[5] <= 1
 R72: - 11.32277806052313 alphas[0] + 23.64033641212048 alphas[1]
   - 55.39416425950258 alphas[2] - 6.646941564979127 alphas[3] - alphas[4]
   + f_var[5] = 0
 R73: - 3725.621915456575 Fraction[5] + Q0_var[5] = 0
 Constr_Left_0_diff_aux_5: Sold_0[5] - Q0_var[5] + Left_0_aux_5 = 0
 Constr_Lost_0_diff_aux_5: Q0_var[5] + Lost_0_aux_5 = 439.4246992254438
 R76: Q0_var[5] + Q_hat_adjusted_5 = 550.2155207808601
 Constr_Q1_plus_left_5: - Left_0[5] - Q1_var[5] + Q1_plus_left[5] = 0
 Constr_Left_1_diff_aux_5: Sold_1[5] - Q1_plus_left[5] + Left_0_aux_5 = 0
 Constr_Lost_1_diff_aux_5: Q1_plus_left[5] + Lost_0_aux_5
   = 110.7883285190089
 Profit_Constraint_5: - 800 Sold_0[5] + 800 Lost_0[5] - 800 Sold_1[5]
   + 200 Left_1[5] + 800 Lost_1[5] + profits_vars[5] = 0
 R81: - 800 Sold_0[5] + 800 Lost_0[5] - 800 Sold_1[5] + 200 Left_1[5]
   + 800 Lost_1[5] + profits_vars[5] = 0
 Fraction_lower_bound_6: Fraction[6] >= 0
 Fraction_upper_bound_6: Fraction[6] <= 1
 R84: - 13.87116844717108 alphas[0] + 21.7147401097066 alphas[1]
   - 51.33119773674819 alphas[2] - 11.21478401499763 alphas[3] - alphas[4]
   + f_var[6] = 0
 R85: - 3725.621915456575 Fraction[6] + Q0_var[6] = 0
 Constr_Left_0_diff_aux_6: Sold_0[6] - Q0_var[6] + Left_0_aux_6 = 0
 Constr_Lost_0_diff_aux_6: Q0_var[6] + Lost_0_aux_6 = 414.4647874540947
 R88: Q0_var[6] + Q_hat_adjusted_6 = 517.1296760658815
 Constr_Q1_plus_left_6: - Left_0[6] - Q1_var[6] + Q1_plus_left[6] = 0
 Constr_Left_1_diff_aux_6: Sold_1[6] - Q1_plus_left[6] + Left_0_aux_6 = 0
 Constr_Lost_1_diff_aux_6: Q1_plus_left[6] + Lost_0_aux_6
   = 102.6623954734367
 Profit_Constraint_6: - 800 Sold_0[6] + 800 Lost_0[6] - 800 Sold_1[6]
   + 200 Left_1[6] + 800 Lost_1[6] + profits_vars[6] = 0
 R93: - 800 Sold_0[6] + 800 Lost_0[6] - 800 Sold_1[6] + 200 Left_1[6]
   + 800 Lost_1[6] + profits_vars[6] = 0
 Fraction_lower_bound_7: Fraction[7] >= 0
 Fraction_upper_bound_7: Fraction[7] <= 1
 R96: - 12.28075166108274 alphas[0] + 28.71705007302431 alphas[1]
   - 81.82007139306633 alphas[2] - 10.77228588604168 alphas[3] - alphas[4]
   + f_var[7] = 0
 R97: - 3725.621915456575 Fraction[7] + Q0_var[7] = 0
 Constr_Left_0_diff_aux_7: Sold_0[7] - Q0_var[7] + Left_0_aux_7 = 0
 Constr_Lost_0_diff_aux_7: Q0_var[7] + Lost_0_aux_7 = 653.7725597306143
 R100: Q0_var[7] + Q_hat_adjusted_7 = 817.4151954905378
 Constr_Q1_plus_left_7: - Left_0[7] - Q1_var[7] + Q1_plus_left[7] = 0
 Constr_Left_1_diff_aux_7: Sold_1[7] - Q1_plus_left[7] + Left_0_aux_7 = 0
 Constr_Lost_1_diff_aux_7: Q1_plus_left[7] + Lost_0_aux_7
   = 163.6401427861327
 Profit_Constraint_7: - 800 Sold_0[7] + 800 Lost_0[7] - 800 Sold_1[7]
   + 200 Left_1[7] + 800 Lost_1[7] + profits_vars[7] = 0
 R105: - 800 Sold_0[7] + 800 Lost_0[7] - 800 Sold_1[7] + 200 Left_1[7]
   + 800 Lost_1[7] + profits_vars[7] = 0
 Fraction_lower_bound_8: Fraction[8] >= 0
 Fraction_upper_bound_8: Fraction[8] <= 1
 R108: - 12.84216974434324 alphas[0] + 16.1505576125996 alphas[1]
   - 32.56627180545316 alphas[2] - 7.378928213745086 alphas[3] - alphas[4]
   + f_var[8] = 0
 R109: - 3725.621915456575 Fraction[8] + Q0_var[8] = 0
 Constr_Left_0_diff_aux_8: Sold_0[8] - Q0_var[8] + Left_0_aux_8 = 0
 Constr_Lost_0_diff_aux_8: Q0_var[8] + Lost_0_aux_8 = 258.8007235013071
 R112: Q0_var[8] + Q_hat_adjusted_8 = 323.9357602399024
 Constr_Q1_plus_left_8: - Left_0[8] - Q1_var[8] + Q1_plus_left[8] = 0
 Constr_Left_1_diff_aux_8: Sold_1[8] - Q1_plus_left[8] + Left_0_aux_8 = 0
 Constr_Lost_1_diff_aux_8: Q1_plus_left[8] + Lost_0_aux_8
   = 65.13254369304477
 Profit_Constraint_8: - 800 Sold_0[8] + 800 Lost_0[8] - 800 Sold_1[8]
   + 200 Left_1[8] + 800 Lost_1[8] + profits_vars[8] = 0
 R117: - 800 Sold_0[8] + 800 Lost_0[8] - 800 Sold_1[8] + 200 Left_1[8]
   + 800 Lost_1[8] + profits_vars[8] = 0
 Fraction_lower_bound_9: Fraction[9] >= 0
 Fraction_upper_bound_9: Fraction[9] <= 1
 R120: - 10.09394900218178 alphas[0] + 18.6679709158685 alphas[1]
   - 50.74700754411094 alphas[2] - 14.34213997924794 alphas[3] - alphas[4]
   + f_var[9] = 0
 R121: - 3725.621915456575 Fraction[9] + Q0_var[9] = 0
 Constr_Left_0_diff_aux_9: Sold_0[9] - Q0_var[9] + Left_0_aux_9 = 0
 Constr_Lost_0_diff_aux_9: Q0_var[9] + Lost_0_aux_9 = 409.7858752208164
 R124: Q0_var[9] + Q_hat_adjusted_9 = 511.2823832952424
 Constr_Q1_plus_left_9: - Left_0[9] - Q1_var[9] + Q1_plus_left[9] = 0
 Constr_Left_1_diff_aux_9: Sold_1[9] - Q1_plus_left[9] + Left_0_aux_9 = 0
 Constr_Lost_1_diff_aux_9: Q1_plus_left[9] + Lost_0_aux_9
   = 101.494015089541
 Profit_Constraint_9: - 800 Sold_0[9] + 800 Lost_0[9] - 800 Sold_1[9]
   + 200 Left_1[9] + 800 Lost_1[9] + profits_vars[9] = 0
 R129: - 800 Sold_0[9] + 800 Lost_0[9] - 800 Sold_1[9] + 200 Left_1[9]
   + 800 Lost_1[9] + profits_vars[9] = 0
 Fraction_lower_bound_10: Fraction[10] >= 0
 Fraction_upper_bound_10: Fraction[10] <= 1
 R132: - 13.08817748537938 alphas[0] + 24.69221018121109 alphas[1]
   - 72.46854687515062 alphas[2] - 11.13965955965896 alphas[3] - alphas[4]
   + f_var[10] = 0
 R133: - 3725.621915456575 Fraction[10] + Q0_var[10] = 0
 Constr_Left_0_diff_aux_10: Sold_0[10] - Q0_var[10] + Left_0_aux_10 = 0
 Constr_Lost_0_diff_aux_10: Q0_var[10] + Lost_0_aux_10 = 573.8441321175674
 R136: Q0_var[10] + Q_hat_adjusted_10 = 718.7837188895861
 Constr_Q1_plus_left_10: - Left_0[10] - Q1_var[10] + Q1_plus_left[10] = 0
 Constr_Left_1_diff_aux_10: Sold_1[10] - Q1_plus_left[10] + Left_0_aux_10
   = 0
 Constr_Lost_1_diff_aux_10: Q1_plus_left[10] + Lost_0_aux_10
   = 144.937093750302
 Profit_Constraint_10: - 800 Sold_0[10] + 800 Lost_0[10] - 800 Sold_1[10]
   + 200 Left_1[10] + 800 Lost_1[10] + profits_vars[10] = 0
 R141: - 800 Sold_0[10] + 800 Lost_0[10] - 800 Sold_1[10] + 200 Left_1[10]
   + 800 Lost_1[10] + profits_vars[10] = 0
 Fraction_lower_bound_11: Fraction[11] >= 0
 Fraction_upper_bound_11: Fraction[11] <= 1
 R144: - 13.06047861361211 alphas[0] + 19.535038930666 alphas[1]
   - 67.41699803342256 alphas[2] - 10.35632803024958 alphas[3] - alphas[4]
   + f_var[11] = 0
 R145: - 3725.621915456575 Fraction[11] + Q0_var[11] = 0
 Constr_Left_0_diff_aux_11: Sold_0[11] - Q0_var[11] + Left_0_aux_11 = 0
 Constr_Lost_0_diff_aux_11: Q0_var[11] + Lost_0_aux_11 = 534.8560746735757
 R148: Q0_var[11] + Q_hat_adjusted_11 = 669.6925637368306
 Constr_Q1_plus_left_11: - Left_0[11] - Q1_var[11] + Q1_plus_left[11] = 0
 Constr_Left_1_diff_aux_11: Sold_1[11] - Q1_plus_left[11] + Left_0_aux_11
   = 0
 Constr_Lost_1_diff_aux_11: Q1_plus_left[11] + Lost_0_aux_11
   = 134.8339960668884
 Profit_Constraint_11: - 800 Sold_0[11] + 800 Lost_0[11] - 800 Sold_1[11]
   + 200 Left_1[11] + 800 Lost_1[11] + profits_vars[11] = 0
 R153: - 800 Sold_0[11] + 800 Lost_0[11] - 800 Sold_1[11] + 200 Left_1[11]
   + 800 Lost_1[11] + profits_vars[11] = 0
 Fraction_lower_bound_12: Fraction[12] >= 0
 Fraction_upper_bound_12: Fraction[12] <= 1
 R156: - 13.08466998437378 alphas[0] + 28.12118978483117 alphas[1]
   - 76.35754243648107 alphas[2] - 10.89909976354571 alphas[3] - alphas[4]
   + f_var[12] = 0
 R157: - 3725.621915456575 Fraction[12] + Q0_var[12] = 0
 Constr_Left_0_diff_aux_12: Sold_0[12] - Q0_var[12] + Left_0_aux_12 = 0
 Constr_Lost_0_diff_aux_12: Q0_var[12] + Lost_0_aux_12 = 607.5999108777356
 R160: Q0_var[12] + Q_hat_adjusted_12 = 760.3174887643182
 Constr_Q1_plus_left_12: - Left_0[12] - Q1_var[12] + Q1_plus_left[12] = 0
 Constr_Left_1_diff_aux_12: Sold_1[12] - Q1_plus_left[12] + Left_0_aux_12
   = 0
 Constr_Lost_1_diff_aux_12: Q1_plus_left[12] + Lost_0_aux_12
   = 152.7150848729622
 Profit_Constraint_12: - 800 Sold_0[12] + 800 Lost_0[12] - 800 Sold_1[12]
   + 200 Left_1[12] + 800 Lost_1[12] + profits_vars[12] = 0
 R165: - 800 Sold_0[12] + 800 Lost_0[12] - 800 Sold_1[12] + 200 Left_1[12]
   + 800 Lost_1[12] + profits_vars[12] = 0
 Fraction_lower_bound_13: Fraction[13] >= 0
 Fraction_upper_bound_13: Fraction[13] <= 1
 R168: - 14.71874039257312 alphas[0] + 18.48107008887641 alphas[1]
   - 57.76287062947319 alphas[2] - 12.3012202951677 alphas[3] - alphas[4]
   + f_var[13] = 0
 R169: - 3725.621915456575 Fraction[13] + Q0_var[13] = 0
 Constr_Left_0_diff_aux_13: Sold_0[13] - Q0_var[13] + Left_0_aux_13 = 0
 Constr_Lost_0_diff_aux_13: Q0_var[13] + Lost_0_aux_13 = 472.1286089814943
 R172: Q0_var[13] + Q_hat_adjusted_13 = 587.6568432831173
 Constr_Q1_plus_left_13: - Left_0[13] - Q1_var[13] + Q1_plus_left[13] = 0
 Constr_Left_1_diff_aux_13: Sold_1[13] - Q1_plus_left[13] + Left_0_aux_13
   = 0
 Constr_Lost_1_diff_aux_13: Q1_plus_left[13] + Lost_0_aux_13
   = 115.5257412587135
 Profit_Constraint_13: - 800 Sold_0[13] + 800 Lost_0[13] - 800 Sold_1[13]
   + 200 Left_1[13] + 800 Lost_1[13] + profits_vars[13] = 0
 R177: - 800 Sold_0[13] + 800 Lost_0[13] - 800 Sold_1[13] + 200 Left_1[13]
   + 800 Lost_1[13] + profits_vars[13] = 0
 Fraction_lower_bound_14: Fraction[14] >= 0
 Fraction_upper_bound_14: Fraction[14] <= 1
 R180: - 13.40910149551742 alphas[0] + 11.41407604847572 alphas[1]
   - 88.67843464581688 alphas[2] - 8.119449954796018 alphas[3] - alphas[4]
   + f_var[14] = 0
 R181: - 3725.621915456575 Fraction[14] + Q0_var[14] = 0
 Constr_Left_0_diff_aux_14: Sold_0[14] - Q0_var[14] + Left_0_aux_14 = 0
 Constr_Lost_0_diff_aux_14: Q0_var[14] + Lost_0_aux_14 = 710.245724504431
 R184: Q0_var[14] + Q_hat_adjusted_14 = 887.6050867861221
 Constr_Q1_plus_left_14: - Left_0[14] - Q1_var[14] + Q1_plus_left[14] = 0
 Constr_Left_1_diff_aux_14: Sold_1[14] - Q1_plus_left[14] + Left_0_aux_14
   = 0
 Constr_Lost_1_diff_aux_14: Q1_plus_left[14] + Lost_0_aux_14
   = 177.3568692919608
 Profit_Constraint_14: - 800 Sold_0[14] + 800 Lost_0[14] - 800 Sold_1[14]
   + 200 Left_1[14] + 800 Lost_1[14] + profits_vars[14] = 0
 R189: - 800 Sold_0[14] + 800 Lost_0[14] - 800 Sold_1[14] + 200 Left_1[14]
   + 800 Lost_1[14] + profits_vars[14] = 0
 Fraction_lower_bound_15: Fraction[15] >= 0
 Fraction_upper_bound_15: Fraction[15] <= 1
 R192: - 11.79753950286893 alphas[0] + 23.62862095097353 alphas[1]
   - 116.3855605923231 alphas[2] - 8.982210622160919 alphas[3] - alphas[4]
   + f_var[15] = 0
 R193: - 3725.621915456575 Fraction[15] + Q0_var[15] = 0
 Constr_Left_0_diff_aux_15: Sold_0[15] - Q0_var[15] + Left_0_aux_15 = 0
 Constr_Lost_0_diff_aux_15: Q0_var[15] + Lost_0_aux_15 = 929.9891278112539
 R196: Q0_var[15] + Q_hat_adjusted_15 = 1162.76274197047
 Constr_Q1_plus_left_15: - Left_0[15] - Q1_var[15] + Q1_plus_left[15] = 0
 Constr_Left_1_diff_aux_15: Sold_1[15] - Q1_plus_left[15] + Left_0_aux_15
   = 0
 Constr_Lost_1_diff_aux_15: Q1_plus_left[15] + Lost_0_aux_15
   = 232.7711211846463
 Profit_Constraint_15: - 800 Sold_0[15] + 800 Lost_0[15] - 800 Sold_1[15]
   + 200 Left_1[15] + 800 Lost_1[15] + profits_vars[15] = 0
 R201: - 800 Sold_0[15] + 800 Lost_0[15] - 800 Sold_1[15] + 200 Left_1[15]
   + 800 Lost_1[15] + profits_vars[15] = 0
 Fraction_lower_bound_16: Fraction[16] >= 0
 Fraction_upper_bound_16: Fraction[16] <= 1
 R204: - 12.18515976899671 alphas[0] + 16.65179240072636 alphas[1]
   - 41.75318559620331 alphas[2] - 7.098437489751221 alphas[3] - alphas[4]
   + f_var[16] = 0
 R205: - 3725.621915456575 Fraction[16] + Q0_var[16] = 0
 Constr_Left_0_diff_aux_16: Sold_0[16] - Q0_var[16] + Left_0_aux_16 = 0
 Constr_Lost_0_diff_aux_16: Q0_var[16] + Lost_0_aux_16 = 332.4735830697661
 R208: Q0_var[16] + Q_hat_adjusted_16 = 415.9824473103827
 Constr_Q1_plus_left_16: - Left_0[16] - Q1_var[16] + Q1_plus_left[16] = 0
 Constr_Left_1_diff_aux_16: Sold_1[16] - Q1_plus_left[16] + Left_0_aux_16
   = 0
 Constr_Lost_1_diff_aux_16: Q1_plus_left[16] + Lost_0_aux_16
   = 83.50637117294735
 Profit_Constraint_16: - 800 Sold_0[16] + 800 Lost_0[16] - 800 Sold_1[16]
   + 200 Left_1[16] + 800 Lost_1[16] + profits_vars[16] = 0
 R213: - 800 Sold_0[16] + 800 Lost_0[16] - 800 Sold_1[16] + 200 Left_1[16]
   + 800 Lost_1[16] + profits_vars[16] = 0
 Fraction_lower_bound_17: Fraction[17] >= 0
 Fraction_upper_bound_17: Fraction[17] <= 1
 R216: - 13.48815597963632 alphas[0] + 27.36404275191216 alphas[1]
   - 81.73791071541142 alphas[2] - 6.861930058803361 alphas[3] - alphas[4]
   + f_var[17] = 0
 R217: - 3725.621915456575 Fraction[17] + Q0_var[17] = 0
 Constr_Left_0_diff_aux_17: Sold_0[17] - Q0_var[17] + Left_0_aux_17 = 0
 Constr_Lost_0_diff_aux_17: Q0_var[17] + Lost_0_aux_17 = 665.1460432763918
 R220: Q0_var[17] + Q_hat_adjusted_17 = 828.6243577060927
 Constr_Q1_plus_left_17: - Left_0[17] - Q1_var[17] + Q1_plus_left[17] = 0
 Constr_Left_1_diff_aux_17: Sold_1[17] - Q1_plus_left[17] + Left_0_aux_17
   = 0
 Constr_Lost_1_diff_aux_17: Q1_plus_left[17] + Lost_0_aux_17
   = 163.4758214308228
 Profit_Constraint_17: - 800 Sold_0[17] + 800 Lost_0[17] - 800 Sold_1[17]
   + 200 Left_1[17] + 800 Lost_1[17] + profits_vars[17] = 0
 R225: - 800 Sold_0[17] + 800 Lost_0[17] - 800 Sold_1[17] + 200 Left_1[17]
   + 800 Lost_1[17] + profits_vars[17] = 0
 Fraction_lower_bound_18: Fraction[18] >= 0
 Fraction_upper_bound_18: Fraction[18] <= 1
 R228: - 10.30112735814635 alphas[0] + 15.67345591762869 alphas[1]
   - 43.20681063451533 alphas[2] - 14.44372389983934 alphas[3] - alphas[4]
   + f_var[18] = 0
 R229: - 3725.621915456575 Fraction[18] + Q0_var[18] = 0
 Constr_Left_0_diff_aux_18: Sold_0[18] - Q0_var[18] + Left_0_aux_18 = 0
 Constr_Lost_0_diff_aux_18: Q0_var[18] + Lost_0_aux_18 = 343.9439564366446
 R232: Q0_var[18] + Q_hat_adjusted_18 = 430.3600707680798
 Constr_Q1_plus_left_18: - Left_0[18] - Q1_var[18] + Q1_plus_left[18] = 0
 Constr_Left_1_diff_aux_18: Sold_1[18] - Q1_plus_left[18] + Left_0_aux_18
   = 0
 Constr_Lost_1_diff_aux_18: Q1_plus_left[18] + Lost_0_aux_18
   = 86.41362109075789
 Profit_Constraint_18: - 800 Sold_0[18] + 800 Lost_0[18] - 800 Sold_1[18]
   + 200 Left_1[18] + 800 Lost_1[18] + profits_vars[18] = 0
 R237: - 800 Sold_0[18] + 800 Lost_0[18] - 800 Sold_1[18] + 200 Left_1[18]
   + 800 Lost_1[18] + profits_vars[18] = 0
 Fraction_lower_bound_19: Fraction[19] >= 0
 Fraction_upper_bound_19: Fraction[19] <= 1
 R240: - 13.33383357722834 alphas[0] + 24.21187814105598 alphas[1]
   - 101.6859681192594 alphas[2] - 12.39550795049288 alphas[3] - alphas[4]
   + f_var[19] = 0
 R241: - 3725.621915456575 Fraction[19] + Q0_var[19] = 0
 Constr_Left_0_diff_aux_19: Sold_0[19] - Q0_var[19] + Left_0_aux_19 = 0
 Constr_Lost_0_diff_aux_19: Q0_var[19] + Lost_0_aux_19 = 815.3538870637935
 R244: Q0_var[19] + Q_hat_adjusted_19 = 1018.72831603994
 Constr_Q1_plus_left_19: - Left_0[19] - Q1_var[19] + Q1_plus_left[19] = 0
 Constr_Left_1_diff_aux_19: Sold_1[19] - Q1_plus_left[19] + Left_0_aux_19
   = 0
 Constr_Lost_1_diff_aux_19: Q1_plus_left[19] + Lost_0_aux_19
   = 203.3719362385187
 Profit_Constraint_19: - 800 Sold_0[19] + 800 Lost_0[19] - 800 Sold_1[19]
   + 200 Left_1[19] + 800 Lost_1[19] + profits_vars[19] = 0
 R249: - 800 Sold_0[19] + 800 Lost_0[19] - 800 Sold_1[19] + 200 Left_1[19]
   + 800 Lost_1[19] + profits_vars[19] = 0
 Fraction_lower_bound_20: Fraction[20] >= 0
 Fraction_upper_bound_20: Fraction[20] <= 1
 R252: - 13.3531893480908 alphas[0] + 26.33617275985766 alphas[1]
   - 69.60597028072938 alphas[2] - 9.904588086175671 alphas[3] - alphas[4]
   + f_var[20] = 0
 R253: - 3725.621915456575 Fraction[20] + Q0_var[20] = 0
 Constr_Left_0_diff_aux_20: Sold_0[20] - Q0_var[20] + Left_0_aux_20 = 0
 Constr_Lost_0_diff_aux_20: Q0_var[20] + Lost_0_aux_20 = 552.6438402183745
 R256: Q0_var[20] + Q_hat_adjusted_20 = 691.8582737815863
 Constr_Q1_plus_left_20: - Left_0[20] - Q1_var[20] + Q1_plus_left[20] = 0
 Constr_Left_1_diff_aux_20: Sold_1[20] - Q1_plus_left[20] + Left_0_aux_20
   = 0
 Constr_Lost_1_diff_aux_20: Q1_plus_left[20] + Lost_0_aux_20
   = 139.2119405614584
 Profit_Constraint_20: - 800 Sold_0[20] + 800 Lost_0[20] - 800 Sold_1[20]
   + 200 Left_1[20] + 800 Lost_1[20] + profits_vars[20] = 0
 R261: - 800 Sold_0[20] + 800 Lost_0[20] - 800 Sold_1[20] + 200 Left_1[20]
   + 800 Lost_1[20] + profits_vars[20] = 0
 Fraction_lower_bound_21: Fraction[21] >= 0
 Fraction_upper_bound_21: Fraction[21] <= 1
 R264: - 11.0519128053692 alphas[0] + 18.26974130379834 alphas[1]
   - 86.54213118585089 alphas[2] - 7.274146279733232 alphas[3] - alphas[4]
   + f_var[21] = 0
 R265: - 3725.621915456575 Fraction[21] + Q0_var[21] = 0
 Constr_Left_0_diff_aux_21: Sold_0[21] - Q0_var[21] + Left_0_aux_21 = 0
 Constr_Lost_0_diff_aux_21: Q0_var[21] + Lost_0_aux_21 = 696.6904528510263
 R268: Q0_var[21] + Q_hat_adjusted_21 = 869.7772081298818
 Constr_Q1_plus_left_21: - Left_0[21] - Q1_var[21] + Q1_plus_left[21] = 0
 Constr_Left_1_diff_aux_21: Sold_1[21] - Q1_plus_left[21] + Left_0_aux_21
   = 0
 Constr_Lost_1_diff_aux_21: Q1_plus_left[21] + Lost_0_aux_21
   = 173.0842623716959
 Profit_Constraint_21: - 800 Sold_0[21] + 800 Lost_0[21] - 800 Sold_1[21]
   + 200 Left_1[21] + 800 Lost_1[21] + profits_vars[21] = 0
 R273: - 800 Sold_0[21] + 800 Lost_0[21] - 800 Sold_1[21] + 200 Left_1[21]
   + 800 Lost_1[21] + profits_vars[21] = 0
 Fraction_lower_bound_22: Fraction[22] >= 0
 Fraction_upper_bound_22: Fraction[22] <= 1
 R276: - 10.64463148827427 alphas[0] + 29.59784907625013 alphas[1]
   - 48.32798362140786 alphas[2] - 7.543564817703929 alphas[3] - alphas[4]
   + f_var[22] = 0
 R277: - 3725.621915456575 Fraction[22] + Q0_var[22] = 0
 Constr_Left_0_diff_aux_22: Sold_0[22] - Q0_var[22] + Left_0_aux_22 = 0
 Constr_Lost_0_diff_aux_22: Q0_var[22] + Lost_0_aux_22 = 384.3453857648469
 R280: Q0_var[22] + Q_hat_adjusted_22 = 481.0038459507478
 Constr_Q1_plus_left_22: - Left_0[22] - Q1_var[22] + Q1_plus_left[22] = 0
 Constr_Left_1_diff_aux_22: Sold_1[22] - Q1_plus_left[22] + Left_0_aux_22
   = 0
 Constr_Lost_1_diff_aux_22: Q1_plus_left[22] + Lost_0_aux_22
   = 96.65596724281588
 Profit_Constraint_22: - 800 Sold_0[22] + 800 Lost_0[22] - 800 Sold_1[22]
   + 200 Left_1[22] + 800 Lost_1[22] + profits_vars[22] = 0
 R285: - 800 Sold_0[22] + 800 Lost_0[22] - 800 Sold_1[22] + 200 Left_1[22]
   + 800 Lost_1[22] + profits_vars[22] = 0
 Fraction_lower_bound_23: Fraction[23] >= 0
 Fraction_upper_bound_23: Fraction[23] <= 1
 R288: - 11.57714175462092 alphas[0] + 13.42119941565274 alphas[1]
   - 44.48477593433773 alphas[2] - 5.580291603238756 alphas[3] - alphas[4]
   + f_var[23] = 0
 R289: - 3725.621915456575 Fraction[23] + Q0_var[23] = 0
 Constr_Left_0_diff_aux_23: Sold_0[23] - Q0_var[23] + Left_0_aux_23 = 0
 Constr_Lost_0_diff_aux_23: Q0_var[23] + Lost_0_aux_23 = 358.6648855771362
 R292: Q0_var[23] + Q_hat_adjusted_23 = 447.6369305960916
 Constr_Q1_plus_left_23: - Left_0[23] - Q1_var[23] + Q1_plus_left[23] = 0
 Constr_Left_1_diff_aux_23: Sold_1[23] - Q1_plus_left[23] + Left_0_aux_23
   = 0
 Constr_Lost_1_diff_aux_23: Q1_plus_left[23] + Lost_0_aux_23
   = 88.96955188184566
 Profit_Constraint_23: - 800 Sold_0[23] + 800 Lost_0[23] - 800 Sold_1[23]
   + 200 Left_1[23] + 800 Lost_1[23] + profits_vars[23] = 0
 R297: - 800 Sold_0[23] + 800 Lost_0[23] - 800 Sold_1[23] + 200 Left_1[23]
   + 800 Lost_1[23] + profits_vars[23] = 0
 Fraction_lower_bound_24: Fraction[24] >= 0
 Fraction_upper_bound_24: Fraction[24] <= 1
 R300: - 11.81855385471311 alphas[0] + 29.90609047614906 alphas[1]
   - 78.80562806489546 alphas[2] - 9.344166255581207 alphas[3] - alphas[4]
   + f_var[24] = 0
 R301: - 3725.621915456575 Fraction[24] + Q0_var[24] = 0
 Constr_Left_0_diff_aux_24: Sold_0[24] - Q0_var[24] + Left_0_aux_24 = 0
 Constr_Lost_0_diff_aux_24: Q0_var[24] + Lost_0_aux_24 = 624.2988442739866
 R304: Q0_var[24] + Q_hat_adjusted_24 = 781.9125933316739
 Constr_Q1_plus_left_24: - Left_0[24] - Q1_var[24] + Q1_plus_left[24] = 0
 Constr_Left_1_diff_aux_24: Sold_1[24] - Q1_plus_left[24] + Left_0_aux_24
   = 0
 Constr_Lost_1_diff_aux_24: Q1_plus_left[24] + Lost_0_aux_24
   = 157.6112561297909
 Profit_Constraint_24: - 800 Sold_0[24] + 800 Lost_0[24] - 800 Sold_1[24]
   + 200 Left_1[24] + 800 Lost_1[24] + profits_vars[24] = 0
 R309: - 800 Sold_0[24] + 800 Lost_0[24] - 800 Sold_1[24] + 200 Left_1[24]
   + 800 Lost_1[24] + profits_vars[24] = 0
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
 Q0_var[0] <= 3726.621915456575
 Q0_var[1] <= 3726.621915456575
 Q0_var[2] <= 3726.621915456575
 Q0_var[3] <= 3726.621915456575
 Q0_var[4] <= 3726.621915456575
 Q0_var[5] <= 3726.621915456575
 Q0_var[6] <= 3726.621915456575
 Q0_var[7] <= 3726.621915456575
 Q0_var[8] <= 3726.621915456575
 Q0_var[9] <= 3726.621915456575
 Q0_var[10] <= 3726.621915456575
 Q0_var[11] <= 3726.621915456575
 Q0_var[12] <= 3726.621915456575
 Q0_var[13] <= 3726.621915456575
 Q0_var[14] <= 3726.621915456575
 Q0_var[15] <= 3726.621915456575
 Q0_var[16] <= 3726.621915456575
 Q0_var[17] <= 3726.621915456575
 Q0_var[18] <= 3726.621915456575
 Q0_var[19] <= 3726.621915456575
 Q0_var[20] <= 3726.621915456575
 Q0_var[21] <= 3726.621915456575
 Q0_var[22] <= 3726.621915456575
 Q0_var[23] <= 3726.621915456575
 Q0_var[24] <= 3726.621915456575
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
 Left_0_aux_0 free
 Lost_0_aux_0 free
 Left_0_aux_0 free
 Lost_0_aux_0 free
 Q_hat_adjusted_0 free
 Left_0_aux_1 free
 Lost_0_aux_1 free
 Left_0_aux_1 free
 Lost_0_aux_1 free
 Q_hat_adjusted_1 free
 Left_0_aux_2 free
 Lost_0_aux_2 free
 Left_0_aux_2 free
 Lost_0_aux_2 free
 Q_hat_adjusted_2 free
 Left_0_aux_3 free
 Lost_0_aux_3 free
 Left_0_aux_3 free
 Lost_0_aux_3 free
 Q_hat_adjusted_3 free
 Left_0_aux_4 free
 Lost_0_aux_4 free
 Left_0_aux_4 free
 Lost_0_aux_4 free
 Q_hat_adjusted_4 free
 Left_0_aux_5 free
 Lost_0_aux_5 free
 Left_0_aux_5 free
 Lost_0_aux_5 free
 Q_hat_adjusted_5 free
 Left_0_aux_6 free
 Lost_0_aux_6 free
 Left_0_aux_6 free
 Lost_0_aux_6 free
 Q_hat_adjusted_6 free
 Left_0_aux_7 free
 Lost_0_aux_7 free
 Left_0_aux_7 free
 Lost_0_aux_7 free
 Q_hat_adjusted_7 free
 Left_0_aux_8 free
 Lost_0_aux_8 free
 Left_0_aux_8 free
 Lost_0_aux_8 free
 Q_hat_adjusted_8 free
 Left_0_aux_9 free
 Lost_0_aux_9 free
 Left_0_aux_9 free
 Lost_0_aux_9 free
 Q_hat_adjusted_9 free
 Left_0_aux_10 free
 Lost_0_aux_10 free
 Left_0_aux_10 free
 Lost_0_aux_10 free
 Q_hat_adjusted_10 free
 Left_0_aux_11 free
 Lost_0_aux_11 free
 Left_0_aux_11 free
 Lost_0_aux_11 free
 Q_hat_adjusted_11 free
 Left_0_aux_12 free
 Lost_0_aux_12 free
 Left_0_aux_12 free
 Lost_0_aux_12 free
 Q_hat_adjusted_12 free
 Left_0_aux_13 free
 Lost_0_aux_13 free
 Left_0_aux_13 free
 Lost_0_aux_13 free
 Q_hat_adjusted_13 free
 Left_0_aux_14 free
 Lost_0_aux_14 free
 Left_0_aux_14 free
 Lost_0_aux_14 free
 Q_hat_adjusted_14 free
 Left_0_aux_15 free
 Lost_0_aux_15 free
 Left_0_aux_15 free
 Lost_0_aux_15 free
 Q_hat_adjusted_15 free
 Left_0_aux_16 free
 Lost_0_aux_16 free
 Left_0_aux_16 free
 Lost_0_aux_16 free
 Q_hat_adjusted_16 free
 Left_0_aux_17 free
 Lost_0_aux_17 free
 Left_0_aux_17 free
 Lost_0_aux_17 free
 Q_hat_adjusted_17 free
 Left_0_aux_18 free
 Lost_0_aux_18 free
 Left_0_aux_18 free
 Lost_0_aux_18 free
 Q_hat_adjusted_18 free
 Left_0_aux_19 free
 Lost_0_aux_19 free
 Left_0_aux_19 free
 Lost_0_aux_19 free
 Q_hat_adjusted_19 free
 Left_0_aux_20 free
 Lost_0_aux_20 free
 Left_0_aux_20 free
 Lost_0_aux_20 free
 Q_hat_adjusted_20 free
 Left_0_aux_21 free
 Lost_0_aux_21 free
 Left_0_aux_21 free
 Lost_0_aux_21 free
 Q_hat_adjusted_21 free
 Left_0_aux_22 free
 Lost_0_aux_22 free
 Left_0_aux_22 free
 Lost_0_aux_22 free
 Q_hat_adjusted_22 free
 Left_0_aux_23 free
 Lost_0_aux_23 free
 Left_0_aux_23 free
 Lost_0_aux_23 free
 Q_hat_adjusted_23 free
 Left_0_aux_24 free
 Lost_0_aux_24 free
 Left_0_aux_24 free
 Lost_0_aux_24 free
 Q_hat_adjusted_24 free
General Constraints
 GC0: Fraction[0] = LOGISTIC ( f_var[0] )
 Constr_Sold_0_min_0: Sold_0[0] = MIN ( Q0_var[0] , 1001.760973706182 )
 Constr_Left_0_max_0: Left_0[0] = MAX ( Left_0_aux_0 , 0 )
 Constr_Lost_0_max_0: Lost_0[0] = MAX ( Lost_0_aux_0 , 0 )
 max_Q1_constr_0: Q1_var[0] = MAX ( Q_hat_adjusted_0 , 0 )
 Constr_Sold_1_min_0: Sold_1[0] = MIN ( Q1_plus_left[0] , 250.5583313943889 )
 Constr_Left_1_max_0: Left_1[0] = MAX ( Left_0_aux_0 , 0 )
 Constr_Lost_1_max_0: Lost_1[0] = MAX ( Lost_0_aux_0 , 0 )
 GC8: Fraction[1] = LOGISTIC ( f_var[1] )
 Constr_Sold_0_min_1: Sold_0[1] = MIN ( Q0_var[1] , 794.8359660065993 )
 Constr_Left_0_max_1: Left_0[1] = MAX ( Left_0_aux_1 , 0 )
 Constr_Lost_0_max_1: Lost_0[1] = MAX ( Lost_0_aux_1 , 0 )
 max_Q1_constr_1: Q1_var[1] = MAX ( Q_hat_adjusted_1 , 0 )
 Constr_Sold_1_min_1: Sold_1[1] = MIN ( Q1_plus_left[1] , 197.4976552775631 )
 Constr_Left_1_max_1: Left_1[1] = MAX ( Left_0_aux_1 , 0 )
 Constr_Lost_1_max_1: Lost_1[1] = MAX ( Lost_0_aux_1 , 0 )
 GC16: Fraction[2] = LOGISTIC ( f_var[2] )
 Constr_Sold_0_min_2: Sold_0[2] = MIN ( Q0_var[2] , 407.1404742176073 )
 Constr_Left_0_max_2: Left_0[2] = MAX ( Left_0_aux_2 , 0 )
 Constr_Lost_0_max_2: Lost_0[2] = MAX ( Lost_0_aux_2 , 0 )
 max_Q1_constr_2: Q1_var[2] = MAX ( Q_hat_adjusted_2 , 0 )
 Constr_Sold_1_min_2: Sold_1[2] = MIN ( Q1_plus_left[2] , 103.1015354227212 )
 Constr_Left_1_max_2: Left_1[2] = MAX ( Left_0_aux_2 , 0 )
 Constr_Lost_1_max_2: Lost_1[2] = MAX ( Lost_0_aux_2 , 0 )
 GC24: Fraction[3] = LOGISTIC ( f_var[3] )
 Constr_Sold_0_min_3: Sold_0[3] = MIN ( Q0_var[3] , 1001.660824741478 )
 Constr_Left_0_max_3: Left_0[3] = MAX ( Left_0_aux_3 , 0 )
 Constr_Lost_0_max_3: Lost_0[3] = MAX ( Lost_0_aux_3 , 0 )
 max_Q1_constr_3: Q1_var[3] = MAX ( Q_hat_adjusted_3 , 0 )
 Constr_Sold_1_min_3: Sold_1[3] = MIN ( Q1_plus_left[3] , 249.4741180977848 )
 Constr_Left_1_max_3: Left_1[3] = MAX ( Left_0_aux_3 , 0 )
 Constr_Lost_1_max_3: Lost_1[3] = MAX ( Lost_0_aux_3 , 0 )
 GC32: Fraction[4] = LOGISTIC ( f_var[4] )
 Constr_Sold_0_min_4: Sold_0[4] = MIN ( Q0_var[4] , 811.2887843628971 )
 Constr_Left_0_max_4: Left_0[4] = MAX ( Left_0_aux_4 , 0 )
 Constr_Lost_0_max_4: Lost_0[4] = MAX ( Lost_0_aux_4 , 0 )
 max_Q1_constr_4: Q1_var[4] = MAX ( Q_hat_adjusted_4 , 0 )
 Constr_Sold_1_min_4: Sold_1[4] = MIN ( Q1_plus_left[4] , 206.1711613540316 )
 Constr_Left_1_max_4: Left_1[4] = MAX ( Left_0_aux_4 , 0 )
 Constr_Lost_1_max_4: Lost_1[4] = MAX ( Lost_0_aux_4 , 0 )
 GC40: Fraction[5] = LOGISTIC ( f_var[5] )
 Constr_Sold_0_min_5: Sold_0[5] = MIN ( Q0_var[5] , 439.4246992254438 )
 Constr_Left_0_max_5: Left_0[5] = MAX ( Left_0_aux_5 , 0 )
 Constr_Lost_0_max_5: Lost_0[5] = MAX ( Lost_0_aux_5 , 0 )
 max_Q1_constr_5: Q1_var[5] = MAX ( Q_hat_adjusted_5 , 0 )
 Constr_Sold_1_min_5: Sold_1[5] = MIN ( Q1_plus_left[5] , 110.7883285190089 )
 Constr_Left_1_max_5: Left_1[5] = MAX ( Left_0_aux_5 , 0 )
 Constr_Lost_1_max_5: Lost_1[5] = MAX ( Lost_0_aux_5 , 0 )
 GC48: Fraction[6] = LOGISTIC ( f_var[6] )
 Constr_Sold_0_min_6: Sold_0[6] = MIN ( Q0_var[6] , 414.4647874540947 )
 Constr_Left_0_max_6: Left_0[6] = MAX ( Left_0_aux_6 , 0 )
 Constr_Lost_0_max_6: Lost_0[6] = MAX ( Lost_0_aux_6 , 0 )
 max_Q1_constr_6: Q1_var[6] = MAX ( Q_hat_adjusted_6 , 0 )
 Constr_Sold_1_min_6: Sold_1[6] = MIN ( Q1_plus_left[6] , 102.6623954734367 )
 Constr_Left_1_max_6: Left_1[6] = MAX ( Left_0_aux_6 , 0 )
 Constr_Lost_1_max_6: Lost_1[6] = MAX ( Lost_0_aux_6 , 0 )
 GC56: Fraction[7] = LOGISTIC ( f_var[7] )
 Constr_Sold_0_min_7: Sold_0[7] = MIN ( Q0_var[7] , 653.7725597306143 )
 Constr_Left_0_max_7: Left_0[7] = MAX ( Left_0_aux_7 , 0 )
 Constr_Lost_0_max_7: Lost_0[7] = MAX ( Lost_0_aux_7 , 0 )
 max_Q1_constr_7: Q1_var[7] = MAX ( Q_hat_adjusted_7 , 0 )
 Constr_Sold_1_min_7: Sold_1[7] = MIN ( Q1_plus_left[7] , 163.6401427861327 )
 Constr_Left_1_max_7: Left_1[7] = MAX ( Left_0_aux_7 , 0 )
 Constr_Lost_1_max_7: Lost_1[7] = MAX ( Lost_0_aux_7 , 0 )
 GC64: Fraction[8] = LOGISTIC ( f_var[8] )
 Constr_Sold_0_min_8: Sold_0[8] = MIN ( Q0_var[8] , 258.8007235013071 )
 Constr_Left_0_max_8: Left_0[8] = MAX ( Left_0_aux_8 , 0 )
 Constr_Lost_0_max_8: Lost_0[8] = MAX ( Lost_0_aux_8 , 0 )
 max_Q1_constr_8: Q1_var[8] = MAX ( Q_hat_adjusted_8 , 0 )
 Constr_Sold_1_min_8: Sold_1[8] = MIN ( Q1_plus_left[8] , 65.13254369304477 )
 Constr_Left_1_max_8: Left_1[8] = MAX ( Left_0_aux_8 , 0 )
 Constr_Lost_1_max_8: Lost_1[8] = MAX ( Lost_0_aux_8 , 0 )
 GC72: Fraction[9] = LOGISTIC ( f_var[9] )
 Constr_Sold_0_min_9: Sold_0[9] = MIN ( Q0_var[9] , 409.7858752208164 )
 Constr_Left_0_max_9: Left_0[9] = MAX ( Left_0_aux_9 , 0 )
 Constr_Lost_0_max_9: Lost_0[9] = MAX ( Lost_0_aux_9 , 0 )
 max_Q1_constr_9: Q1_var[9] = MAX ( Q_hat_adjusted_9 , 0 )
 Constr_Sold_1_min_9: Sold_1[9] = MIN ( Q1_plus_left[9] , 101.494015089541 )
 Constr_Left_1_max_9: Left_1[9] = MAX ( Left_0_aux_9 , 0 )
 Constr_Lost_1_max_9: Lost_1[9] = MAX ( Lost_0_aux_9 , 0 )
 GC80: Fraction[10] = LOGISTIC ( f_var[10] )
 Constr_Sold_0_min_10: Sold_0[10] = MIN ( Q0_var[10] , 573.8441321175674 )
 Constr_Left_0_max_10: Left_0[10] = MAX ( Left_0_aux_10 , 0 )
 Constr_Lost_0_max_10: Lost_0[10] = MAX ( Lost_0_aux_10 , 0 )
 max_Q1_constr_10: Q1_var[10] = MAX ( Q_hat_adjusted_10 , 0 )
 Constr_Sold_1_min_10: Sold_1[10] = MIN ( Q1_plus_left[10]
   , 144.937093750302 )
 Constr_Left_1_max_10: Left_1[10] = MAX ( Left_0_aux_10 , 0 )
 Constr_Lost_1_max_10: Lost_1[10] = MAX ( Lost_0_aux_10 , 0 )
 GC88: Fraction[11] = LOGISTIC ( f_var[11] )
 Constr_Sold_0_min_11: Sold_0[11] = MIN ( Q0_var[11] , 534.8560746735757 )
 Constr_Left_0_max_11: Left_0[11] = MAX ( Left_0_aux_11 , 0 )
 Constr_Lost_0_max_11: Lost_0[11] = MAX ( Lost_0_aux_11 , 0 )
 max_Q1_constr_11: Q1_var[11] = MAX ( Q_hat_adjusted_11 , 0 )
 Constr_Sold_1_min_11: Sold_1[11] = MIN ( Q1_plus_left[11]
   , 134.8339960668884 )
 Constr_Left_1_max_11: Left_1[11] = MAX ( Left_0_aux_11 , 0 )
 Constr_Lost_1_max_11: Lost_1[11] = MAX ( Lost_0_aux_11 , 0 )
 GC96: Fraction[12] = LOGISTIC ( f_var[12] )
 Constr_Sold_0_min_12: Sold_0[12] = MIN ( Q0_var[12] , 607.5999108777356 )
 Constr_Left_0_max_12: Left_0[12] = MAX ( Left_0_aux_12 , 0 )
 Constr_Lost_0_max_12: Lost_0[12] = MAX ( Lost_0_aux_12 , 0 )
 max_Q1_constr_12: Q1_var[12] = MAX ( Q_hat_adjusted_12 , 0 )
 Constr_Sold_1_min_12: Sold_1[12] = MIN ( Q1_plus_left[12]
   , 152.7150848729622 )
 Constr_Left_1_max_12: Left_1[12] = MAX ( Left_0_aux_12 , 0 )
 Constr_Lost_1_max_12: Lost_1[12] = MAX ( Lost_0_aux_12 , 0 )
 GC104: Fraction[13] = LOGISTIC ( f_var[13] )
 Constr_Sold_0_min_13: Sold_0[13] = MIN ( Q0_var[13] , 472.1286089814943 )
 Constr_Left_0_max_13: Left_0[13] = MAX ( Left_0_aux_13 , 0 )
 Constr_Lost_0_max_13: Lost_0[13] = MAX ( Lost_0_aux_13 , 0 )
 max_Q1_constr_13: Q1_var[13] = MAX ( Q_hat_adjusted_13 , 0 )
 Constr_Sold_1_min_13: Sold_1[13] = MIN ( Q1_plus_left[13]
   , 115.5257412587135 )
 Constr_Left_1_max_13: Left_1[13] = MAX ( Left_0_aux_13 , 0 )
 Constr_Lost_1_max_13: Lost_1[13] = MAX ( Lost_0_aux_13 , 0 )
 GC112: Fraction[14] = LOGISTIC ( f_var[14] )
 Constr_Sold_0_min_14: Sold_0[14] = MIN ( Q0_var[14] , 710.245724504431 )
 Constr_Left_0_max_14: Left_0[14] = MAX ( Left_0_aux_14 , 0 )
 Constr_Lost_0_max_14: Lost_0[14] = MAX ( Lost_0_aux_14 , 0 )
 max_Q1_constr_14: Q1_var[14] = MAX ( Q_hat_adjusted_14 , 0 )
 Constr_Sold_1_min_14: Sold_1[14] = MIN ( Q1_plus_left[14]
   , 177.3568692919608 )
 Constr_Left_1_max_14: Left_1[14] = MAX ( Left_0_aux_14 , 0 )
 Constr_Lost_1_max_14: Lost_1[14] = MAX ( Lost_0_aux_14 , 0 )
 GC120: Fraction[15] = LOGISTIC ( f_var[15] )
 Constr_Sold_0_min_15: Sold_0[15] = MIN ( Q0_var[15] , 929.9891278112539 )
 Constr_Left_0_max_15: Left_0[15] = MAX ( Left_0_aux_15 , 0 )
 Constr_Lost_0_max_15: Lost_0[15] = MAX ( Lost_0_aux_15 , 0 )
 max_Q1_constr_15: Q1_var[15] = MAX ( Q_hat_adjusted_15 , 0 )
 Constr_Sold_1_min_15: Sold_1[15] = MIN ( Q1_plus_left[15]
   , 232.7711211846463 )
 Constr_Left_1_max_15: Left_1[15] = MAX ( Left_0_aux_15 , 0 )
 Constr_Lost_1_max_15: Lost_1[15] = MAX ( Lost_0_aux_15 , 0 )
 GC128: Fraction[16] = LOGISTIC ( f_var[16] )
 Constr_Sold_0_min_16: Sold_0[16] = MIN ( Q0_var[16] , 332.4735830697661 )
 Constr_Left_0_max_16: Left_0[16] = MAX ( Left_0_aux_16 , 0 )
 Constr_Lost_0_max_16: Lost_0[16] = MAX ( Lost_0_aux_16 , 0 )
 max_Q1_constr_16: Q1_var[16] = MAX ( Q_hat_adjusted_16 , 0 )
 Constr_Sold_1_min_16: Sold_1[16] = MIN ( Q1_plus_left[16]
   , 83.50637117294735 )
 Constr_Left_1_max_16: Left_1[16] = MAX ( Left_0_aux_16 , 0 )
 Constr_Lost_1_max_16: Lost_1[16] = MAX ( Lost_0_aux_16 , 0 )
 GC136: Fraction[17] = LOGISTIC ( f_var[17] )
 Constr_Sold_0_min_17: Sold_0[17] = MIN ( Q0_var[17] , 665.1460432763918 )
 Constr_Left_0_max_17: Left_0[17] = MAX ( Left_0_aux_17 , 0 )
 Constr_Lost_0_max_17: Lost_0[17] = MAX ( Lost_0_aux_17 , 0 )
 max_Q1_constr_17: Q1_var[17] = MAX ( Q_hat_adjusted_17 , 0 )
 Constr_Sold_1_min_17: Sold_1[17] = MIN ( Q1_plus_left[17]
   , 163.4758214308228 )
 Constr_Left_1_max_17: Left_1[17] = MAX ( Left_0_aux_17 , 0 )
 Constr_Lost_1_max_17: Lost_1[17] = MAX ( Lost_0_aux_17 , 0 )
 GC144: Fraction[18] = LOGISTIC ( f_var[18] )
 Constr_Sold_0_min_18: Sold_0[18] = MIN ( Q0_var[18] , 343.9439564366446 )
 Constr_Left_0_max_18: Left_0[18] = MAX ( Left_0_aux_18 , 0 )
 Constr_Lost_0_max_18: Lost_0[18] = MAX ( Lost_0_aux_18 , 0 )
 max_Q1_constr_18: Q1_var[18] = MAX ( Q_hat_adjusted_18 , 0 )
 Constr_Sold_1_min_18: Sold_1[18] = MIN ( Q1_plus_left[18]
   , 86.41362109075789 )
 Constr_Left_1_max_18: Left_1[18] = MAX ( Left_0_aux_18 , 0 )
 Constr_Lost_1_max_18: Lost_1[18] = MAX ( Lost_0_aux_18 , 0 )
 GC152: Fraction[19] = LOGISTIC ( f_var[19] )
 Constr_Sold_0_min_19: Sold_0[19] = MIN ( Q0_var[19] , 815.3538870637935 )
 Constr_Left_0_max_19: Left_0[19] = MAX ( Left_0_aux_19 , 0 )
 Constr_Lost_0_max_19: Lost_0[19] = MAX ( Lost_0_aux_19 , 0 )
 max_Q1_constr_19: Q1_var[19] = MAX ( Q_hat_adjusted_19 , 0 )
 Constr_Sold_1_min_19: Sold_1[19] = MIN ( Q1_plus_left[19]
   , 203.3719362385187 )
 Constr_Left_1_max_19: Left_1[19] = MAX ( Left_0_aux_19 , 0 )
 Constr_Lost_1_max_19: Lost_1[19] = MAX ( Lost_0_aux_19 , 0 )
 GC160: Fraction[20] = LOGISTIC ( f_var[20] )
 Constr_Sold_0_min_20: Sold_0[20] = MIN ( Q0_var[20] , 552.6438402183745 )
 Constr_Left_0_max_20: Left_0[20] = MAX ( Left_0_aux_20 , 0 )
 Constr_Lost_0_max_20: Lost_0[20] = MAX ( Lost_0_aux_20 , 0 )
 max_Q1_constr_20: Q1_var[20] = MAX ( Q_hat_adjusted_20 , 0 )
 Constr_Sold_1_min_20: Sold_1[20] = MIN ( Q1_plus_left[20]
   , 139.2119405614584 )
 Constr_Left_1_max_20: Left_1[20] = MAX ( Left_0_aux_20 , 0 )
 Constr_Lost_1_max_20: Lost_1[20] = MAX ( Lost_0_aux_20 , 0 )
 GC168: Fraction[21] = LOGISTIC ( f_var[21] )
 Constr_Sold_0_min_21: Sold_0[21] = MIN ( Q0_var[21] , 696.6904528510263 )
 Constr_Left_0_max_21: Left_0[21] = MAX ( Left_0_aux_21 , 0 )
 Constr_Lost_0_max_21: Lost_0[21] = MAX ( Lost_0_aux_21 , 0 )
 max_Q1_constr_21: Q1_var[21] = MAX ( Q_hat_adjusted_21 , 0 )
 Constr_Sold_1_min_21: Sold_1[21] = MIN ( Q1_plus_left[21]
   , 173.0842623716959 )
 Constr_Left_1_max_21: Left_1[21] = MAX ( Left_0_aux_21 , 0 )
 Constr_Lost_1_max_21: Lost_1[21] = MAX ( Lost_0_aux_21 , 0 )
 GC176: Fraction[22] = LOGISTIC ( f_var[22] )
 Constr_Sold_0_min_22: Sold_0[22] = MIN ( Q0_var[22] , 384.3453857648469 )
 Constr_Left_0_max_22: Left_0[22] = MAX ( Left_0_aux_22 , 0 )
 Constr_Lost_0_max_22: Lost_0[22] = MAX ( Lost_0_aux_22 , 0 )
 max_Q1_constr_22: Q1_var[22] = MAX ( Q_hat_adjusted_22 , 0 )
 Constr_Sold_1_min_22: Sold_1[22] = MIN ( Q1_plus_left[22]
   , 96.65596724281588 )
 Constr_Left_1_max_22: Left_1[22] = MAX ( Left_0_aux_22 , 0 )
 Constr_Lost_1_max_22: Lost_1[22] = MAX ( Lost_0_aux_22 , 0 )
 GC184: Fraction[23] = LOGISTIC ( f_var[23] )
 Constr_Sold_0_min_23: Sold_0[23] = MIN ( Q0_var[23] , 358.6648855771362 )
 Constr_Left_0_max_23: Left_0[23] = MAX ( Left_0_aux_23 , 0 )
 Constr_Lost_0_max_23: Lost_0[23] = MAX ( Lost_0_aux_23 , 0 )
 max_Q1_constr_23: Q1_var[23] = MAX ( Q_hat_adjusted_23 , 0 )
 Constr_Sold_1_min_23: Sold_1[23] = MIN ( Q1_plus_left[23]
   , 88.96955188184566 )
 Constr_Left_1_max_23: Left_1[23] = MAX ( Left_0_aux_23 , 0 )
 Constr_Lost_1_max_23: Lost_1[23] = MAX ( Lost_0_aux_23 , 0 )
 GC192: Fraction[24] = LOGISTIC ( f_var[24] )
 Constr_Sold_0_min_24: Sold_0[24] = MIN ( Q0_var[24] , 624.2988442739866 )
 Constr_Left_0_max_24: Left_0[24] = MAX ( Left_0_aux_24 , 0 )
 Constr_Lost_0_max_24: Lost_0[24] = MAX ( Lost_0_aux_24 , 0 )
 max_Q1_constr_24: Q1_var[24] = MAX ( Q_hat_adjusted_24 , 0 )
 Constr_Sold_1_min_24: Sold_1[24] = MIN ( Q1_plus_left[24]
   , 157.6112561297909 )
 Constr_Left_1_max_24: Left_1[24] = MAX ( Left_0_aux_24 , 0 )
 Constr_Lost_1_max_24: Lost_1[24] = MAX ( Lost_0_aux_24 , 0 )
End
