\ Model profit_maximization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 0.1 abs_alpha[0] - 0.1 abs_alpha[1] - 0.1 abs_alpha[2]
   - 0.1 abs_alpha[3] + profits_vars[0] + profits_vars[1] + profits_vars[2]
   + profits_vars[3] + profits_vars[4] + profits_vars[5] + profits_vars[6]
   + profits_vars[7] + profits_vars[8] + profits_vars[9] + profits_vars[10]
   + profits_vars[11] + profits_vars[12] + profits_vars[13]
   + profits_vars[14]
Subject To
 R0: - alphas[0] + abs_alpha[0] >= 0
 R1: alphas[0] + abs_alpha[0] >= 0
 R2: - alphas[1] + abs_alpha[1] >= 0
 R3: alphas[1] + abs_alpha[1] >= 0
 R4: - alphas[2] + abs_alpha[2] >= 0
 R5: alphas[2] + abs_alpha[2] >= 0
 R6: - alphas[3] + abs_alpha[3] >= 0
 R7: alphas[3] + abs_alpha[3] >= 0
 Fraction_lower_bound_0: Fraction[0] >= 0
 Fraction_upper_bound_0: Fraction[0] <= 1
 R10: - 250 alphas[0] - 6.31793916729104 alphas[2] - alphas[3] + f_var[0]
   = 0
 R11: - 2439.950183572415 Fraction[0] + Q0_var[0] = 0
 Constr_Left_0_diff_aux_0: Sold_0[0] - Q0_var[0] + Left_0_aux_0 = 0
 Constr_Lost_0_diff_aux_0: Q0_var[0] + Lost_0_aux_0 = 1889.150713386467
 R14: Q0_var[0] + Q_hat_adjusted_0 = 2475.409928016043
 Constr_Q1_plus_left_0: - Left_0[0] - Q1_var[0] + Q1_plus_left[0] = 0
 Constr_Left_1_diff_aux_0: Sold_1[0] - Q1_plus_left[0] + Left_0_aux_0 = 0
 Constr_Lost_1_diff_aux_0: Q1_plus_left[0] + Lost_0_aux_0
   = 590.3950440322278
 Profit_Constraint_0: - 600 Sold_0[0] + 400 Left_0[0] + 600 Lost_0[0]
   - 600 Sold_1[0] + 400 Left_1[0] + 600 Lost_1[0] + profits_vars[0] = 0
 Fraction_lower_bound_1: Fraction[1] >= 0
 Fraction_upper_bound_1: Fraction[1] <= 1
 R21: - 250 alphas[0] - 6.22075028232861 alphas[2] - alphas[3] + f_var[1]
   = 0
 R22: - 2439.950183572415 Fraction[1] + Q0_var[1] = 0
 Constr_Left_0_diff_aux_1: Sold_0[1] - Q0_var[1] + Left_0_aux_1 = 0
 Constr_Lost_0_diff_aux_1: Q0_var[1] + Lost_0_aux_1 = 1835.18728544783
 R25: Q0_var[1] + Q_hat_adjusted_1 = 2423.926536378417
 Constr_Q1_plus_left_1: - Left_0[1] - Q1_var[1] + Q1_plus_left[1] = 0
 Constr_Left_1_diff_aux_1: Sold_1[1] - Q1_plus_left[1] + Left_0_aux_1 = 0
 Constr_Lost_1_diff_aux_1: Q1_plus_left[1] + Lost_0_aux_1
   = 583.4702403198478
 Profit_Constraint_1: - 600 Sold_0[1] + 400 Left_0[1] + 600 Lost_0[1]
   - 600 Sold_1[1] + 400 Left_1[1] + 600 Lost_1[1] + profits_vars[1] = 0
 Fraction_lower_bound_2: Fraction[2] >= 0
 Fraction_upper_bound_2: Fraction[2] <= 1
 R32: - 250 alphas[0] - 7.312659178930547 alphas[2] - alphas[3] + f_var[2]
   = 0
 R33: - 2439.950183572415 Fraction[2] + Q0_var[2] = 0
 Constr_Left_0_diff_aux_2: Sold_0[2] - Q0_var[2] + Left_0_aux_2 = 0
 Constr_Lost_0_diff_aux_2: Q0_var[2] + Lost_0_aux_2 = 1801.156674136158
 R36: Q0_var[2] + Q_hat_adjusted_2 = 2394.352304305168
 Constr_Q1_plus_left_2: - Left_0[2] - Q1_var[2] + Q1_plus_left[2] = 0
 Constr_Left_1_diff_aux_2: Sold_1[2] - Q1_plus_left[2] + Left_0_aux_2 = 0
 Constr_Lost_1_diff_aux_2: Q1_plus_left[2] + Lost_0_aux_2
   = 592.3267974263101
 Profit_Constraint_2: - 600 Sold_0[2] + 400 Left_0[2] + 600 Lost_0[2]
   - 600 Sold_1[2] + 400 Left_1[2] + 600 Lost_1[2] + profits_vars[2] = 0
 Fraction_lower_bound_3: Fraction[3] >= 0
 Fraction_upper_bound_3: Fraction[3] <= 1
 R43: - 250 alphas[0] - 6.276365902697227 alphas[2] - alphas[3] + f_var[3]
   = 0
 R44: - 2439.950183572415 Fraction[3] + Q0_var[3] = 0
 Constr_Left_0_diff_aux_3: Sold_0[3] - Q0_var[3] + Left_0_aux_3 = 0
 Constr_Lost_0_diff_aux_3: Q0_var[3] + Lost_0_aux_3 = 1893.180636194625
 R47: Q0_var[3] + Q_hat_adjusted_3 = 2497.761702881949
 Constr_Q1_plus_left_3: - Left_0[3] - Q1_var[3] + Q1_plus_left[3] = 0
 Constr_Left_1_diff_aux_3: Sold_1[3] - Q1_plus_left[3] + Left_0_aux_3 = 0
 Constr_Lost_1_diff_aux_3: Q1_plus_left[3] + Lost_0_aux_3
   = 600.8663995998115
 Profit_Constraint_3: - 600 Sold_0[3] + 400 Left_0[3] + 600 Lost_0[3]
   - 600 Sold_1[3] + 400 Left_1[3] + 600 Lost_1[3] + profits_vars[3] = 0
 Fraction_lower_bound_4: Fraction[4] >= 0
 Fraction_upper_bound_4: Fraction[4] <= 1
 R54: - 250 alphas[0] - 6.393164723360107 alphas[2] - alphas[3] + f_var[4]
   = 0
 R55: - 2439.950183572415 Fraction[4] + Q0_var[4] = 0
 Constr_Left_0_diff_aux_4: Sold_0[4] - Q0_var[4] + Left_0_aux_4 = 0
 Constr_Lost_0_diff_aux_4: Q0_var[4] + Lost_0_aux_4 = 1889.694426349152
 R58: Q0_var[4] + Q_hat_adjusted_4 = 2489.961909036796
 Constr_Q1_plus_left_4: - Left_0[4] - Q1_var[4] + Q1_plus_left[4] = 0
 Constr_Left_1_diff_aux_4: Sold_1[4] - Q1_plus_left[4] + Left_0_aux_4 = 0
 Constr_Lost_1_diff_aux_4: Q1_plus_left[4] + Lost_0_aux_4
   = 596.6155579941149
 Profit_Constraint_4: - 600 Sold_0[4] + 400 Left_0[4] + 600 Lost_0[4]
   - 600 Sold_1[4] + 400 Left_1[4] + 600 Lost_1[4] + profits_vars[4] = 0
 Fraction_lower_bound_5: Fraction[5] >= 0
 Fraction_upper_bound_5: Fraction[5] <= 1
 R65: - 250 alphas[0] - 6.737450341321928 alphas[2] - alphas[3] + f_var[5]
   = 0
 R66: - 2439.950183572415 Fraction[5] + Q0_var[5] = 0
 Constr_Left_0_diff_aux_5: Sold_0[5] - Q0_var[5] + Left_0_aux_5 = 0
 Constr_Lost_0_diff_aux_5: Q0_var[5] + Lost_0_aux_5 = 1897.365600268182
 R69: Q0_var[5] + Q_hat_adjusted_5 = 2491.537628239855
 Constr_Q1_plus_left_5: - Left_0[5] - Q1_var[5] + Q1_plus_left[5] = 0
 Constr_Left_1_diff_aux_5: Sold_1[5] - Q1_plus_left[5] + Left_0_aux_5 = 0
 Constr_Lost_1_diff_aux_5: Q1_plus_left[5] + Lost_0_aux_5
   = 582.7917280813149
 Profit_Constraint_5: - 600 Sold_0[5] + 400 Left_0[5] + 600 Lost_0[5]
   - 600 Sold_1[5] + 400 Left_1[5] + 600 Lost_1[5] + profits_vars[5] = 0
 Fraction_lower_bound_6: Fraction[6] >= 0
 Fraction_upper_bound_6: Fraction[6] <= 1
 R76: - 250 alphas[0] - 7.64198645969587 alphas[2] - alphas[3] + f_var[6]
   = 0
 R77: - 2439.950183572415 Fraction[6] + Q0_var[6] = 0
 Constr_Left_0_diff_aux_6: Sold_0[6] - Q0_var[6] + Left_0_aux_6 = 0
 Constr_Lost_0_diff_aux_6: Q0_var[6] + Lost_0_aux_6 = 1782.905233698554
 R80: Q0_var[6] + Q_hat_adjusted_6 = 2383.159241920079
 Constr_Q1_plus_left_6: - Left_0[6] - Q1_var[6] + Q1_plus_left[6] = 0
 Constr_Left_1_diff_aux_6: Sold_1[6] - Q1_plus_left[6] + Left_0_aux_6 = 0
 Constr_Lost_1_diff_aux_6: Q1_plus_left[6] + Lost_0_aux_6
   = 597.6652818524468
 Profit_Constraint_6: - 600 Sold_0[6] + 400 Left_0[6] + 600 Lost_0[6]
   - 600 Sold_1[6] + 400 Left_1[6] + 600 Lost_1[6] + profits_vars[6] = 0
 Fraction_lower_bound_7: Fraction[7] >= 0
 Fraction_upper_bound_7: Fraction[7] <= 1
 R87: - 250 alphas[0] - 6.194202551586122 alphas[2] - alphas[3] + f_var[7]
   = 0
 R88: - 2439.950183572415 Fraction[7] + Q0_var[7] = 0
 Constr_Left_0_diff_aux_7: Sold_0[7] - Q0_var[7] + Left_0_aux_7 = 0
 Constr_Lost_0_diff_aux_7: Q0_var[7] + Lost_0_aux_7 = 1910.774032032563
 R91: Q0_var[7] + Q_hat_adjusted_7 = 2507.701151944827
 Constr_Q1_plus_left_7: - Left_0[7] - Q1_var[7] + Q1_plus_left[7] = 0
 Constr_Left_1_diff_aux_7: Sold_1[7] - Q1_plus_left[7] + Left_0_aux_7 = 0
 Constr_Lost_1_diff_aux_7: Q1_plus_left[7] + Lost_0_aux_7
   = 597.3152682281212
 Profit_Constraint_7: - 600 Sold_0[7] + 400 Left_0[7] + 600 Lost_0[7]
   - 600 Sold_1[7] + 400 Left_1[7] + 600 Lost_1[7] + profits_vars[7] = 0
 Fraction_lower_bound_8: Fraction[8] >= 0
 Fraction_upper_bound_8: Fraction[8] <= 1
 R98: - 250 alphas[0] - 7.675889814997608 alphas[2] - alphas[3] + f_var[8]
   = 0
 R99: - 2439.950183572415 Fraction[8] + Q0_var[8] = 0
 Constr_Left_0_diff_aux_8: Sold_0[8] - Q0_var[8] + Left_0_aux_8 = 0
 Constr_Lost_0_diff_aux_8: Q0_var[8] + Lost_0_aux_8 = 1743.976219300634
 R102: Q0_var[8] + Q_hat_adjusted_8 = 2342.219218785049
 Constr_Q1_plus_left_8: - Left_0[8] - Q1_var[8] + Q1_plus_left[8] = 0
 Constr_Left_1_diff_aux_8: Sold_1[8] - Q1_plus_left[8] + Left_0_aux_8 = 0
 Constr_Lost_1_diff_aux_8: Q1_plus_left[8] + Lost_0_aux_8
   = 605.8058975036208
 Profit_Constraint_8: - 600 Sold_0[8] + 400 Left_0[8] + 600 Lost_0[8]
   - 600 Sold_1[8] + 400 Left_1[8] + 600 Lost_1[8] + profits_vars[8] = 0
 Fraction_lower_bound_9: Fraction[9] >= 0
 Fraction_upper_bound_9: Fraction[9] <= 1
 R109: - 250 alphas[0] - 6.192196815787926 alphas[2] - alphas[3] + f_var[9]
   = 0
 R110: - 2439.950183572415 Fraction[9] + Q0_var[9] = 0
 Constr_Left_0_diff_aux_9: Sold_0[9] - Q0_var[9] + Left_0_aux_9 = 0
 Constr_Lost_0_diff_aux_9: Q0_var[9] + Lost_0_aux_9 = 1911.576876663822
 R113: Q0_var[9] + Q_hat_adjusted_9 = 2528.995929645437
 Constr_Q1_plus_left_9: - Left_0[9] - Q1_var[9] + Q1_plus_left[9] = 0
 Constr_Left_1_diff_aux_9: Sold_1[9] - Q1_plus_left[9] + Left_0_aux_9 = 0
 Constr_Lost_1_diff_aux_9: Q1_plus_left[9] + Lost_0_aux_9
   = 616.6878588153063
 Profit_Constraint_9: - 600 Sold_0[9] + 400 Left_0[9] + 600 Lost_0[9]
   - 600 Sold_1[9] + 400 Left_1[9] + 600 Lost_1[9] + profits_vars[9] = 0
 Fraction_lower_bound_10: Fraction[10] >= 0
 Fraction_upper_bound_10: Fraction[10] <= 1
 R120: - 250 alphas[0] - 7.952918930026792 alphas[2] - alphas[3]
   + f_var[10] = 0
 R121: - 2439.950183572415 Fraction[10] + Q0_var[10] = 0
 Constr_Left_0_diff_aux_10: Sold_0[10] - Q0_var[10] + Left_0_aux_10 = 0
 Constr_Lost_0_diff_aux_10: Q0_var[10] + Lost_0_aux_10 = 1728.858235777891
 R124: Q0_var[10] + Q_hat_adjusted_10 = 2327.190573960247
 Constr_Q1_plus_left_10: - Left_0[10] - Q1_var[10] + Q1_plus_left[10] = 0
 Constr_Left_1_diff_aux_10: Sold_1[10] - Q1_plus_left[10] + Left_0_aux_10
   = 0
 Constr_Lost_1_diff_aux_10: Q1_plus_left[10] + Lost_0_aux_10
   = 590.2720353669572
 Profit_Constraint_10: - 600 Sold_0[10] + 400 Left_0[10] + 600 Lost_0[10]
   - 600 Sold_1[10] + 400 Left_1[10] + 600 Lost_1[10] + profits_vars[10]
   = 0
 Fraction_lower_bound_11: Fraction[11] >= 0
 Fraction_upper_bound_11: Fraction[11] <= 1
 R131: - 250 alphas[0] - 6.937302403295403 alphas[2] - alphas[3]
   + f_var[11] = 0
 R132: - 2439.950183572415 Fraction[11] + Q0_var[11] = 0
 Constr_Left_0_diff_aux_11: Sold_0[11] - Q0_var[11] + Left_0_aux_11 = 0
 Constr_Lost_0_diff_aux_11: Q0_var[11] + Lost_0_aux_11 = 1807.946987145903
 R135: Q0_var[11] + Q_hat_adjusted_11 = 2403.096429310032
 Constr_Q1_plus_left_11: - Left_0[11] - Q1_var[11] + Q1_plus_left[11] = 0
 Constr_Left_1_diff_aux_11: Sold_1[11] - Q1_plus_left[11] + Left_0_aux_11
   = 0
 Constr_Lost_1_diff_aux_11: Q1_plus_left[11] + Lost_0_aux_11
   = 598.3537961750475
 Profit_Constraint_11: - 600 Sold_0[11] + 400 Left_0[11] + 600 Lost_0[11]
   - 600 Sold_1[11] + 400 Left_1[11] + 600 Lost_1[11] + profits_vars[11]
   = 0
 Fraction_lower_bound_12: Fraction[12] >= 0
 Fraction_upper_bound_12: Fraction[12] <= 1
 R142: - 250 alphas[0] - 7.953522176380674 alphas[2] - alphas[3]
   + f_var[12] = 0
 R143: - 2439.950183572415 Fraction[12] + Q0_var[12] = 0
 Constr_Left_0_diff_aux_12: Sold_0[12] - Q0_var[12] + Left_0_aux_12 = 0
 Constr_Lost_0_diff_aux_12: Q0_var[12] + Lost_0_aux_12 = 1726.506594945682
 R146: Q0_var[12] + Q_hat_adjusted_12 = 2320.506676023226
 Constr_Q1_plus_left_12: - Left_0[12] - Q1_var[12] + Q1_plus_left[12] = 0
 Constr_Left_1_diff_aux_12: Sold_1[12] - Q1_plus_left[12] + Left_0_aux_12
   = 0
 Constr_Lost_1_diff_aux_12: Q1_plus_left[12] + Lost_0_aux_12
   = 592.6292536240547
 Profit_Constraint_12: - 600 Sold_0[12] + 400 Left_0[12] + 600 Lost_0[12]
   - 600 Sold_1[12] + 400 Left_1[12] + 600 Lost_1[12] + profits_vars[12]
   = 0
 Fraction_lower_bound_13: Fraction[13] >= 0
 Fraction_upper_bound_13: Fraction[13] <= 1
 R153: - 250 alphas[0] - 7.209691039490092 alphas[2] - alphas[3]
   + f_var[13] = 0
 R154: - 2439.950183572415 Fraction[13] + Q0_var[13] = 0
 Constr_Left_0_diff_aux_13: Sold_0[13] - Q0_var[13] + Left_0_aux_13 = 0
 Constr_Lost_0_diff_aux_13: Q0_var[13] + Lost_0_aux_13 = 1831.902247074759
 R157: Q0_var[13] + Q_hat_adjusted_13 = 2415.466958544016
 Constr_Q1_plus_left_13: - Left_0[13] - Q1_var[13] + Q1_plus_left[13] = 0
 Constr_Left_1_diff_aux_13: Sold_1[13] - Q1_plus_left[13] + Left_0_aux_13
   = 0
 Constr_Lost_1_diff_aux_13: Q1_plus_left[13] + Lost_0_aux_13
   = 593.8322029234319
 Profit_Constraint_13: - 600 Sold_0[13] + 400 Left_0[13] + 600 Lost_0[13]
   - 600 Sold_1[13] + 400 Left_1[13] + 600 Lost_1[13] + profits_vars[13]
   = 0
 Fraction_lower_bound_14: Fraction[14] >= 0
 Fraction_upper_bound_14: Fraction[14] <= 1
 R164: - 250 alphas[0] - 7.478527158796603 alphas[2] - alphas[3]
   + f_var[14] = 0
 R165: - 2439.950183572415 Fraction[14] + Q0_var[14] = 0
 Constr_Left_0_diff_aux_14: Sold_0[14] - Q0_var[14] + Left_0_aux_14 = 0
 Constr_Lost_0_diff_aux_14: Q0_var[14] + Lost_0_aux_14 = 1772.757867318636
 R168: Q0_var[14] + Q_hat_adjusted_14 = 2351.904326524474
 Constr_Q1_plus_left_14: - Left_0[14] - Q1_var[14] + Q1_plus_left[14] = 0
 Constr_Left_1_diff_aux_14: Sold_1[14] - Q1_plus_left[14] + Left_0_aux_14
   = 0
 Constr_Lost_1_diff_aux_14: Q1_plus_left[14] + Lost_0_aux_14
   = 567.4260634088218
 Profit_Constraint_14: - 600 Sold_0[14] + 400 Left_0[14] + 600 Lost_0[14]
   - 600 Sold_1[14] + 400 Left_1[14] + 600 Lost_1[14] + profits_vars[14]
   = 0
Bounds
 alphas[0] free
 alphas[1] free
 alphas[2] free
 alphas[3] free
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
 Q0_var[0] <= 2440.950183572415
 Q0_var[1] <= 2440.950183572415
 Q0_var[2] <= 2440.950183572415
 Q0_var[3] <= 2440.950183572415
 Q0_var[4] <= 2440.950183572415
 Q0_var[5] <= 2440.950183572415
 Q0_var[6] <= 2440.950183572415
 Q0_var[7] <= 2440.950183572415
 Q0_var[8] <= 2440.950183572415
 Q0_var[9] <= 2440.950183572415
 Q0_var[10] <= 2440.950183572415
 Q0_var[11] <= 2440.950183572415
 Q0_var[12] <= 2440.950183572415
 Q0_var[13] <= 2440.950183572415
 Q0_var[14] <= 2440.950183572415
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
General Constraints
 GC0: ( FuncNonlinear=1 ) Fraction[0] = LOGISTIC ( f_var[0] )
 Constr_Sold_0_min_0: Sold_0[0] = MIN ( Q0_var[0] , 1889.150713386467 )
 Constr_Left_0_max_0: Left_0[0] = MAX ( Left_0_aux_0 , 0 )
 Constr_Lost_0_max_0: Lost_0[0] = MAX ( Lost_0_aux_0 , 0 )
 max_Q1_constr_0: Q1_var[0] = MAX ( Q_hat_adjusted_0 , 0 )
 Constr_Sold_1_min_0: Sold_1[0] = MIN ( Q1_plus_left[0] , 590.3950440322278 )
 Constr_Left_1_max_0: Left_1[0] = MAX ( Left_0_aux_0 , 0 )
 Constr_Lost_1_max_0: Lost_1[0] = MAX ( Lost_0_aux_0 , 0 )
 GC8: ( FuncNonlinear=1 ) Fraction[1] = LOGISTIC ( f_var[1] )
 Constr_Sold_0_min_1: Sold_0[1] = MIN ( Q0_var[1] , 1835.18728544783 )
 Constr_Left_0_max_1: Left_0[1] = MAX ( Left_0_aux_1 , 0 )
 Constr_Lost_0_max_1: Lost_0[1] = MAX ( Lost_0_aux_1 , 0 )
 max_Q1_constr_1: Q1_var[1] = MAX ( Q_hat_adjusted_1 , 0 )
 Constr_Sold_1_min_1: Sold_1[1] = MIN ( Q1_plus_left[1] , 583.4702403198478 )
 Constr_Left_1_max_1: Left_1[1] = MAX ( Left_0_aux_1 , 0 )
 Constr_Lost_1_max_1: Lost_1[1] = MAX ( Lost_0_aux_1 , 0 )
 GC16: ( FuncNonlinear=1 ) Fraction[2] = LOGISTIC ( f_var[2] )
 Constr_Sold_0_min_2: Sold_0[2] = MIN ( Q0_var[2] , 1801.156674136158 )
 Constr_Left_0_max_2: Left_0[2] = MAX ( Left_0_aux_2 , 0 )
 Constr_Lost_0_max_2: Lost_0[2] = MAX ( Lost_0_aux_2 , 0 )
 max_Q1_constr_2: Q1_var[2] = MAX ( Q_hat_adjusted_2 , 0 )
 Constr_Sold_1_min_2: Sold_1[2] = MIN ( Q1_plus_left[2] , 592.3267974263101 )
 Constr_Left_1_max_2: Left_1[2] = MAX ( Left_0_aux_2 , 0 )
 Constr_Lost_1_max_2: Lost_1[2] = MAX ( Lost_0_aux_2 , 0 )
 GC24: ( FuncNonlinear=1 ) Fraction[3] = LOGISTIC ( f_var[3] )
 Constr_Sold_0_min_3: Sold_0[3] = MIN ( Q0_var[3] , 1893.180636194625 )
 Constr_Left_0_max_3: Left_0[3] = MAX ( Left_0_aux_3 , 0 )
 Constr_Lost_0_max_3: Lost_0[3] = MAX ( Lost_0_aux_3 , 0 )
 max_Q1_constr_3: Q1_var[3] = MAX ( Q_hat_adjusted_3 , 0 )
 Constr_Sold_1_min_3: Sold_1[3] = MIN ( Q1_plus_left[3] , 600.8663995998115 )
 Constr_Left_1_max_3: Left_1[3] = MAX ( Left_0_aux_3 , 0 )
 Constr_Lost_1_max_3: Lost_1[3] = MAX ( Lost_0_aux_3 , 0 )
 GC32: ( FuncNonlinear=1 ) Fraction[4] = LOGISTIC ( f_var[4] )
 Constr_Sold_0_min_4: Sold_0[4] = MIN ( Q0_var[4] , 1889.694426349152 )
 Constr_Left_0_max_4: Left_0[4] = MAX ( Left_0_aux_4 , 0 )
 Constr_Lost_0_max_4: Lost_0[4] = MAX ( Lost_0_aux_4 , 0 )
 max_Q1_constr_4: Q1_var[4] = MAX ( Q_hat_adjusted_4 , 0 )
 Constr_Sold_1_min_4: Sold_1[4] = MIN ( Q1_plus_left[4] , 596.6155579941149 )
 Constr_Left_1_max_4: Left_1[4] = MAX ( Left_0_aux_4 , 0 )
 Constr_Lost_1_max_4: Lost_1[4] = MAX ( Lost_0_aux_4 , 0 )
 GC40: ( FuncNonlinear=1 ) Fraction[5] = LOGISTIC ( f_var[5] )
 Constr_Sold_0_min_5: Sold_0[5] = MIN ( Q0_var[5] , 1897.365600268182 )
 Constr_Left_0_max_5: Left_0[5] = MAX ( Left_0_aux_5 , 0 )
 Constr_Lost_0_max_5: Lost_0[5] = MAX ( Lost_0_aux_5 , 0 )
 max_Q1_constr_5: Q1_var[5] = MAX ( Q_hat_adjusted_5 , 0 )
 Constr_Sold_1_min_5: Sold_1[5] = MIN ( Q1_plus_left[5] , 582.7917280813149 )
 Constr_Left_1_max_5: Left_1[5] = MAX ( Left_0_aux_5 , 0 )
 Constr_Lost_1_max_5: Lost_1[5] = MAX ( Lost_0_aux_5 , 0 )
 GC48: ( FuncNonlinear=1 ) Fraction[6] = LOGISTIC ( f_var[6] )
 Constr_Sold_0_min_6: Sold_0[6] = MIN ( Q0_var[6] , 1782.905233698554 )
 Constr_Left_0_max_6: Left_0[6] = MAX ( Left_0_aux_6 , 0 )
 Constr_Lost_0_max_6: Lost_0[6] = MAX ( Lost_0_aux_6 , 0 )
 max_Q1_constr_6: Q1_var[6] = MAX ( Q_hat_adjusted_6 , 0 )
 Constr_Sold_1_min_6: Sold_1[6] = MIN ( Q1_plus_left[6] , 597.6652818524468 )
 Constr_Left_1_max_6: Left_1[6] = MAX ( Left_0_aux_6 , 0 )
 Constr_Lost_1_max_6: Lost_1[6] = MAX ( Lost_0_aux_6 , 0 )
 GC56: ( FuncNonlinear=1 ) Fraction[7] = LOGISTIC ( f_var[7] )
 Constr_Sold_0_min_7: Sold_0[7] = MIN ( Q0_var[7] , 1910.774032032563 )
 Constr_Left_0_max_7: Left_0[7] = MAX ( Left_0_aux_7 , 0 )
 Constr_Lost_0_max_7: Lost_0[7] = MAX ( Lost_0_aux_7 , 0 )
 max_Q1_constr_7: Q1_var[7] = MAX ( Q_hat_adjusted_7 , 0 )
 Constr_Sold_1_min_7: Sold_1[7] = MIN ( Q1_plus_left[7] , 597.3152682281212 )
 Constr_Left_1_max_7: Left_1[7] = MAX ( Left_0_aux_7 , 0 )
 Constr_Lost_1_max_7: Lost_1[7] = MAX ( Lost_0_aux_7 , 0 )
 GC64: ( FuncNonlinear=1 ) Fraction[8] = LOGISTIC ( f_var[8] )
 Constr_Sold_0_min_8: Sold_0[8] = MIN ( Q0_var[8] , 1743.976219300634 )
 Constr_Left_0_max_8: Left_0[8] = MAX ( Left_0_aux_8 , 0 )
 Constr_Lost_0_max_8: Lost_0[8] = MAX ( Lost_0_aux_8 , 0 )
 max_Q1_constr_8: Q1_var[8] = MAX ( Q_hat_adjusted_8 , 0 )
 Constr_Sold_1_min_8: Sold_1[8] = MIN ( Q1_plus_left[8] , 605.8058975036208 )
 Constr_Left_1_max_8: Left_1[8] = MAX ( Left_0_aux_8 , 0 )
 Constr_Lost_1_max_8: Lost_1[8] = MAX ( Lost_0_aux_8 , 0 )
 GC72: ( FuncNonlinear=1 ) Fraction[9] = LOGISTIC ( f_var[9] )
 Constr_Sold_0_min_9: Sold_0[9] = MIN ( Q0_var[9] , 1911.576876663822 )
 Constr_Left_0_max_9: Left_0[9] = MAX ( Left_0_aux_9 , 0 )
 Constr_Lost_0_max_9: Lost_0[9] = MAX ( Lost_0_aux_9 , 0 )
 max_Q1_constr_9: Q1_var[9] = MAX ( Q_hat_adjusted_9 , 0 )
 Constr_Sold_1_min_9: Sold_1[9] = MIN ( Q1_plus_left[9] , 616.6878588153063 )
 Constr_Left_1_max_9: Left_1[9] = MAX ( Left_0_aux_9 , 0 )
 Constr_Lost_1_max_9: Lost_1[9] = MAX ( Lost_0_aux_9 , 0 )
 GC80: ( FuncNonlinear=1 ) Fraction[10] = LOGISTIC ( f_var[10] )
 Constr_Sold_0_min_10: Sold_0[10] = MIN ( Q0_var[10] , 1728.858235777891 )
 Constr_Left_0_max_10: Left_0[10] = MAX ( Left_0_aux_10 , 0 )
 Constr_Lost_0_max_10: Lost_0[10] = MAX ( Lost_0_aux_10 , 0 )
 max_Q1_constr_10: Q1_var[10] = MAX ( Q_hat_adjusted_10 , 0 )
 Constr_Sold_1_min_10: Sold_1[10] = MIN ( Q1_plus_left[10]
   , 590.2720353669572 )
 Constr_Left_1_max_10: Left_1[10] = MAX ( Left_0_aux_10 , 0 )
 Constr_Lost_1_max_10: Lost_1[10] = MAX ( Lost_0_aux_10 , 0 )
 GC88: ( FuncNonlinear=1 ) Fraction[11] = LOGISTIC ( f_var[11] )
 Constr_Sold_0_min_11: Sold_0[11] = MIN ( Q0_var[11] , 1807.946987145903 )
 Constr_Left_0_max_11: Left_0[11] = MAX ( Left_0_aux_11 , 0 )
 Constr_Lost_0_max_11: Lost_0[11] = MAX ( Lost_0_aux_11 , 0 )
 max_Q1_constr_11: Q1_var[11] = MAX ( Q_hat_adjusted_11 , 0 )
 Constr_Sold_1_min_11: Sold_1[11] = MIN ( Q1_plus_left[11]
   , 598.3537961750475 )
 Constr_Left_1_max_11: Left_1[11] = MAX ( Left_0_aux_11 , 0 )
 Constr_Lost_1_max_11: Lost_1[11] = MAX ( Lost_0_aux_11 , 0 )
 GC96: ( FuncNonlinear=1 ) Fraction[12] = LOGISTIC ( f_var[12] )
 Constr_Sold_0_min_12: Sold_0[12] = MIN ( Q0_var[12] , 1726.506594945682 )
 Constr_Left_0_max_12: Left_0[12] = MAX ( Left_0_aux_12 , 0 )
 Constr_Lost_0_max_12: Lost_0[12] = MAX ( Lost_0_aux_12 , 0 )
 max_Q1_constr_12: Q1_var[12] = MAX ( Q_hat_adjusted_12 , 0 )
 Constr_Sold_1_min_12: Sold_1[12] = MIN ( Q1_plus_left[12]
   , 592.6292536240547 )
 Constr_Left_1_max_12: Left_1[12] = MAX ( Left_0_aux_12 , 0 )
 Constr_Lost_1_max_12: Lost_1[12] = MAX ( Lost_0_aux_12 , 0 )
 GC104: ( FuncNonlinear=1 ) Fraction[13] = LOGISTIC ( f_var[13] )
 Constr_Sold_0_min_13: Sold_0[13] = MIN ( Q0_var[13] , 1831.902247074759 )
 Constr_Left_0_max_13: Left_0[13] = MAX ( Left_0_aux_13 , 0 )
 Constr_Lost_0_max_13: Lost_0[13] = MAX ( Lost_0_aux_13 , 0 )
 max_Q1_constr_13: Q1_var[13] = MAX ( Q_hat_adjusted_13 , 0 )
 Constr_Sold_1_min_13: Sold_1[13] = MIN ( Q1_plus_left[13]
   , 593.8322029234319 )
 Constr_Left_1_max_13: Left_1[13] = MAX ( Left_0_aux_13 , 0 )
 Constr_Lost_1_max_13: Lost_1[13] = MAX ( Lost_0_aux_13 , 0 )
 GC112: ( FuncNonlinear=1 ) Fraction[14] = LOGISTIC ( f_var[14] )
 Constr_Sold_0_min_14: Sold_0[14] = MIN ( Q0_var[14] , 1772.757867318636 )
 Constr_Left_0_max_14: Left_0[14] = MAX ( Left_0_aux_14 , 0 )
 Constr_Lost_0_max_14: Lost_0[14] = MAX ( Lost_0_aux_14 , 0 )
 max_Q1_constr_14: Q1_var[14] = MAX ( Q_hat_adjusted_14 , 0 )
 Constr_Sold_1_min_14: Sold_1[14] = MIN ( Q1_plus_left[14]
   , 567.4260634088218 )
 Constr_Left_1_max_14: Left_1[14] = MAX ( Left_0_aux_14 , 0 )
 Constr_Lost_1_max_14: Lost_1[14] = MAX ( Lost_0_aux_14 , 0 )
End
