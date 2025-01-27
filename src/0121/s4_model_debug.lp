\ Model profit_maximization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0 Holding_Cost_0[0] + 0 Holding_Cost_0[1] + 0 Holding_Cost_0[2]
   + 0 Holding_Cost_0[3] + 0 Holding_Cost_0[4] + 0 Holding_Cost_0[5]
   + 0 Holding_Cost_0[6] + 0 Holding_Cost_0[7] + 0 Holding_Cost_0[8]
   + 0 Holding_Cost_0[9] + 0 Holding_Cost_1[0] + 0 Holding_Cost_1[1]
   + 0 Holding_Cost_1[2] + 0 Holding_Cost_1[3] + 0 Holding_Cost_1[4]
   + 0 Holding_Cost_1[5] + 0 Holding_Cost_1[6] + 0 Holding_Cost_1[7]
   + 0 Holding_Cost_1[8] + 0 Holding_Cost_1[9] + profits_vars[0]
   + profits_vars[1] + profits_vars[2] + profits_vars[3] + profits_vars[4]
   + profits_vars[5] + profits_vars[6] + profits_vars[7] + profits_vars[8]
   + profits_vars[9] + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
Subject To
 R0: 0.3629370792343273 alphas[0] - 1.380816439025236 alphas[1]
   + 0.1456017588547462 alphas[2] + 0.6508025075686793 alphas[3]
   - alphas[4] + f_var[0] = 0
 Q0_upper_bound_0: - 1453.252517908159 Fraction_for_second_order_amount[0]
   + Q0_var[0] = 0
 tau_computation_with_f_0_0: 0.3629370792343273 betas[0,0]
   - 1.380816439025236 betas[0,1] + 0.1456017588547462 betas[0,2]
   + 0.6508025075686793 betas[0,3] + tau[0,0] = 0
 tau_computation_with_f_0_1: 0.3629370792343273 betas[1,0]
   - 1.380816439025236 betas[1,1] + 0.1456017588547462 betas[1,2]
   + 0.6508025075686793 betas[1,3] + tau[0,1] = 0
 tau_computation_with_f_0_2: 0.3629370792343273 betas[2,0]
   - 1.380816439025236 betas[2,1] + 0.1456017588547462 betas[2,2]
   + 0.6508025075686793 betas[2,3] + tau[0,2] = 0
 tau_computation_with_f_0_3: 0.3629370792343273 betas[3,0]
   - 1.380816439025236 betas[3,1] + 0.1456017588547462 betas[3,2]
   + 0.6508025075686793 betas[3,3] + tau[0,3] = 0
 tau_computation_with_f_0_4: 0.3629370792343273 betas[4,0]
   - 1.380816439025236 betas[4,1] + 0.1456017588547462 betas[4,2]
   + 0.6508025075686793 betas[4,3] + tau[0,4] = 0
 tau_computation_with_f_0_5: 0.3629370792343273 betas[5,0]
   - 1.380816439025236 betas[5,1] + 0.1456017588547462 betas[5,2]
   + 0.6508025075686793 betas[5,3] + tau[0,5] = 0
 tau_computation_with_f_0_6: 0.3629370792343273 betas[6,0]
   - 1.380816439025236 betas[6,1] + 0.1456017588547462 betas[6,2]
   + 0.6508025075686793 betas[6,3] + tau[0,6] = 0
 tau_computation_with_f_0_7: 0.3629370792343273 betas[7,0]
   - 1.380816439025236 betas[7,1] + 0.1456017588547462 betas[7,2]
   + 0.6508025075686793 betas[7,3] + tau[0,7] = 0
 sum_r_0: r[0,0] + r[0,1] + r[0,2] + r[0,3] + r[0,4] + r[0,5] + r[0,6]
   + r[0,7] = 1
 Ensure_only_one_R_true_0: R[0,0] + R[0,1] + R[0,2] + R[0,3] + R[0,4]
   + R[0,5] + R[0,6] + R[0,7] = 1
 Define_Q_hat_0: - 891.2939989566571 R[0,0] - 1290.737878201308 R[0,1]
   - 1230.96250114189 R[0,2] - 1228.401690341659 R[0,3]
   - 1349.172380180563 R[0,4] - 1349.172351192387 R[0,5]
   - 1315.437030263039 R[0,6] - 1348.747552561108 R[0,7] + Q_hat[0] = 0
 Adjust_Q_hat_0: Q0_var[0] - Q_hat[0] + Q_hat_adjusted[0] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_0: Total_Demand_Up_to_K_minus_1[0]
   - 54.46201239340417 R[0,0] - 194.3293744992587 R[0,1]
   - 194.3293744992587 R[0,2] - 334.2063493663247 R[0,3]
   - 712.9968195759043 R[0,4] - 852.8737947193508 R[0,5]
   - 983.7976638321614 R[0,6] - 1068.993602307392 R[0,7] = 0
 Constr_Total_Demand_from_K_to_T_0: Total_Demand_from_k_to_T[0]
   - 1294.285540200882 R[0,0] - 1154.418178095028 R[0,1]
   - 1154.418178095028 R[0,2] - 1014.541203227962 R[0,3]
   - 635.7507330183821 R[0,4] - 495.8737578749356 R[0,5]
   - 364.9498887621249 R[0,6] - 279.7539502868945 R[0,7] = 0
 Constr_Left_0_diff_aux_0: Sold_0[0] - Q0_var[0] + Left_0_aux_0 = 0
 Constr_Lost_0_diff_aux_0: - Total_Demand_Up_to_K_minus_1[0] + Q0_var[0]
   + Lost_0_aux_0 = 0
 Constr_Q1_plus_left_0: - Left_0[0] + Q1_plus_left[0] - Q1_var[0] = 0
 Constr_Left_1_diff_aux_0: Sold_1[0] - Q1_plus_left[0] + Left_0_aux_0 = 0
 Constr_Lost_1_diff_aux_0: - Total_Demand_from_k_to_T[0] + Q1_plus_left[0]
   + Lost_0_aux_0 = 0
 Profit_Constraint_0: - 500 Sold_0[0] + 500 Lost_0[0] - 500 Sold_1[0]
   + 500 Left_1[0] + 500 Lost_1[0] + profits_vars[0] = 0
 R22: - 0.5389518359619722 alphas[0] - 0.7587744334102131 alphas[1]
   - 0.2125425940893099 alphas[2] + 0.8179904522965766 alphas[3]
   - alphas[4] + f_var[1] = 0
 Q0_upper_bound_1: - 1453.252517908159 Fraction_for_second_order_amount[1]
   + Q0_var[1] = 0
 tau_computation_with_f_1_0: - 0.5389518359619722 betas[0,0]
   - 0.7587744334102131 betas[0,1] - 0.2125425940893099 betas[0,2]
   + 0.8179904522965766 betas[0,3] + tau[1,0] = 0
 tau_computation_with_f_1_1: - 0.5389518359619722 betas[1,0]
   - 0.7587744334102131 betas[1,1] - 0.2125425940893099 betas[1,2]
   + 0.8179904522965766 betas[1,3] + tau[1,1] = 0
 tau_computation_with_f_1_2: - 0.5389518359619722 betas[2,0]
   - 0.7587744334102131 betas[2,1] - 0.2125425940893099 betas[2,2]
   + 0.8179904522965766 betas[2,3] + tau[1,2] = 0
 tau_computation_with_f_1_3: - 0.5389518359619722 betas[3,0]
   - 0.7587744334102131 betas[3,1] - 0.2125425940893099 betas[3,2]
   + 0.8179904522965766 betas[3,3] + tau[1,3] = 0
 tau_computation_with_f_1_4: - 0.5389518359619722 betas[4,0]
   - 0.7587744334102131 betas[4,1] - 0.2125425940893099 betas[4,2]
   + 0.8179904522965766 betas[4,3] + tau[1,4] = 0
 tau_computation_with_f_1_5: - 0.5389518359619722 betas[5,0]
   - 0.7587744334102131 betas[5,1] - 0.2125425940893099 betas[5,2]
   + 0.8179904522965766 betas[5,3] + tau[1,5] = 0
 tau_computation_with_f_1_6: - 0.5389518359619722 betas[6,0]
   - 0.7587744334102131 betas[6,1] - 0.2125425940893099 betas[6,2]
   + 0.8179904522965766 betas[6,3] + tau[1,6] = 0
 tau_computation_with_f_1_7: - 0.5389518359619722 betas[7,0]
   - 0.7587744334102131 betas[7,1] - 0.2125425940893099 betas[7,2]
   + 0.8179904522965766 betas[7,3] + tau[1,7] = 0
 sum_r_1: r[1,0] + r[1,1] + r[1,2] + r[1,3] + r[1,4] + r[1,5] + r[1,6]
   + r[1,7] = 1
 Ensure_only_one_R_true_1: R[1,0] + R[1,1] + R[1,2] + R[1,3] + R[1,4]
   + R[1,5] + R[1,6] + R[1,7] = 1
 Define_Q_hat_1: - 1308.724726204294 R[1,0] - 1534.242846461023 R[1,1]
   - 1358.429320801451 R[1,2] - 1359.744399822718 R[1,3]
   - 1340.150880863648 R[1,4] - 1340.150896078258 R[1,5]
   - 1389.568799356635 R[1,6] - 1370.612111769512 R[1,7] + Q_hat[1] = 0
 Adjust_Q_hat_1: Q0_var[1] - Q_hat[1] + Q_hat_adjusted[1] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_1: Total_Demand_Up_to_K_minus_1[1]
   - 160.7867847726715 R[1,0] - 320.0438095906127 R[1,1]
   - 385.3425574445226 R[1,2] - 544.6005457291369 R[1,3]
   - 678.4913052155408 R[1,4] - 837.7492936653762 R[1,5]
   - 1031.794815905025 R[1,6] - 1052.096134848146 R[1,7] = 0
 Constr_Total_Demand_from_K_to_T_1: Total_Demand_from_k_to_T[1]
   - 1209.825326975145 R[1,0] - 1050.568302157204 R[1,1]
   - 985.2695543032937 R[1,2] - 826.0115660186794 R[1,3]
   - 692.1208065322755 R[1,4] - 532.8628180824402 R[1,5]
   - 338.8172958427911 R[1,6] - 318.5159768996707 R[1,7] = 0
 Constr_Left_0_diff_aux_1: Sold_0[1] - Q0_var[1] + Left_0_aux_1 = 0
 Constr_Lost_0_diff_aux_1: - Total_Demand_Up_to_K_minus_1[1] + Q0_var[1]
   + Lost_0_aux_1 = 0
 Constr_Q1_plus_left_1: - Left_0[1] + Q1_plus_left[1] - Q1_var[1] = 0
 Constr_Left_1_diff_aux_1: Sold_1[1] - Q1_plus_left[1] + Left_0_aux_1 = 0
 Constr_Lost_1_diff_aux_1: - Total_Demand_from_k_to_T[1] + Q1_plus_left[1]
   + Lost_0_aux_1 = 0
 Profit_Constraint_1: - 500 Sold_0[1] + 500 Lost_0[1] - 500 Sold_1[1]
   + 500 Left_1[1] + 500 Lost_1[1] + profits_vars[1] = 0
 R44: 0.0704860532973403 alphas[0] + 0.4116864373552839 alphas[1]
   - 1.416454709959565 alphas[2] - 1.060351956133714 alphas[3] - alphas[4]
   + f_var[2] = 0
 Q0_upper_bound_2: - 1453.252517908159 Fraction_for_second_order_amount[2]
   + Q0_var[2] = 0
 tau_computation_with_f_2_0: 0.0704860532973403 betas[0,0]
   + 0.4116864373552839 betas[0,1] - 1.416454709959565 betas[0,2]
   - 1.060351956133714 betas[0,3] + tau[2,0] = 0
 tau_computation_with_f_2_1: 0.0704860532973403 betas[1,0]
   + 0.4116864373552839 betas[1,1] - 1.416454709959565 betas[1,2]
   - 1.060351956133714 betas[1,3] + tau[2,1] = 0
 tau_computation_with_f_2_2: 0.0704860532973403 betas[2,0]
   + 0.4116864373552839 betas[2,1] - 1.416454709959565 betas[2,2]
   - 1.060351956133714 betas[2,3] + tau[2,2] = 0
 tau_computation_with_f_2_3: 0.0704860532973403 betas[3,0]
   + 0.4116864373552839 betas[3,1] - 1.416454709959565 betas[3,2]
   - 1.060351956133714 betas[3,3] + tau[2,3] = 0
 tau_computation_with_f_2_4: 0.0704860532973403 betas[4,0]
   + 0.4116864373552839 betas[4,1] - 1.416454709959565 betas[4,2]
   - 1.060351956133714 betas[4,3] + tau[2,4] = 0
 tau_computation_with_f_2_5: 0.0704860532973403 betas[5,0]
   + 0.4116864373552839 betas[5,1] - 1.416454709959565 betas[5,2]
   - 1.060351956133714 betas[5,3] + tau[2,5] = 0
 tau_computation_with_f_2_6: 0.0704860532973403 betas[6,0]
   + 0.4116864373552839 betas[6,1] - 1.416454709959565 betas[6,2]
   - 1.060351956133714 betas[6,3] + tau[2,6] = 0
 tau_computation_with_f_2_7: 0.0704860532973403 betas[7,0]
   + 0.4116864373552839 betas[7,1] - 1.416454709959565 betas[7,2]
   - 1.060351956133714 betas[7,3] + tau[2,7] = 0
 sum_r_2: r[2,0] + r[2,1] + r[2,2] + r[2,3] + r[2,4] + r[2,5] + r[2,6]
   + r[2,7] = 1
 Ensure_only_one_R_true_2: R[2,0] + R[2,1] + R[2,2] + R[2,3] + R[2,4]
   + R[2,5] + R[2,6] + R[2,7] = 1
 Define_Q_hat_2: - 2330.254303335292 R[2,0] - 2291.220035242185 R[2,1]
   - 2108.798715654455 R[2,2] - 2117.231285059415 R[2,3]
   - 2069.624954603227 R[2,4] - 2069.624938761026 R[2,5]
   - 1981.371584906666 R[2,6] - 1981.600919045505 R[2,7] + Q_hat[2] = 0
 Adjust_Q_hat_2: Q0_var[2] - Q_hat[2] + Q_hat_adjusted[2] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_2: Total_Demand_Up_to_K_minus_1[2]
   - 420.9830105831127 R[2,0] - 645.3908129396151 R[2,1]
   - 957.1143474123051 R[2,2] - 1181.522146325861 R[2,3]
   - 1181.522146325861 R[2,4] - 1405.929945307678 R[2,5]
   - 1434.881134747542 R[2,6] - 1532.785321129661 R[2,7] = 0
 Constr_Total_Demand_from_K_to_T_2: Total_Demand_from_k_to_T[2]
   - 1560.617908510181 R[2,0] - 1336.210106153678 R[2,1]
   - 1024.486571680988 R[2,2] - 800.0787727674319 R[2,3]
   - 800.0787727674319 R[2,4] - 575.6709737856157 R[2,5]
   - 546.7197843457517 R[2,6] - 448.8155979636325 R[2,7] = 0
 Constr_Left_0_diff_aux_2: Sold_0[2] - Q0_var[2] + Left_0_aux_2 = 0
 Constr_Lost_0_diff_aux_2: - Total_Demand_Up_to_K_minus_1[2] + Q0_var[2]
   + Lost_0_aux_2 = 0
 Constr_Q1_plus_left_2: - Left_0[2] + Q1_plus_left[2] - Q1_var[2] = 0
 Constr_Left_1_diff_aux_2: Sold_1[2] - Q1_plus_left[2] + Left_0_aux_2 = 0
 Constr_Lost_1_diff_aux_2: - Total_Demand_from_k_to_T[2] + Q1_plus_left[2]
   + Lost_0_aux_2 = 0
 Profit_Constraint_2: - 500 Sold_0[2] + 500 Lost_0[2] - 500 Sold_1[2]
   + 500 Left_1[2] + 500 Lost_1[2] + profits_vars[2] = 0
 R66: 0.3842425308077874 alphas[0] - 0.6942014022851634 alphas[1]
   + 1.528221828396284 alphas[2] + 0.7223183878646142 alphas[3] - alphas[4]
   + f_var[3] = 0
 Q0_upper_bound_3: - 1453.252517908159 Fraction_for_second_order_amount[3]
   + Q0_var[3] = 0
 tau_computation_with_f_3_0: 0.3842425308077874 betas[0,0]
   - 0.6942014022851634 betas[0,1] + 1.528221828396284 betas[0,2]
   + 0.7223183878646142 betas[0,3] + tau[3,0] = 0
 tau_computation_with_f_3_1: 0.3842425308077874 betas[1,0]
   - 0.6942014022851634 betas[1,1] + 1.528221828396284 betas[1,2]
   + 0.7223183878646142 betas[1,3] + tau[3,1] = 0
 tau_computation_with_f_3_2: 0.3842425308077874 betas[2,0]
   - 0.6942014022851634 betas[2,1] + 1.528221828396284 betas[2,2]
   + 0.7223183878646142 betas[2,3] + tau[3,2] = 0
 tau_computation_with_f_3_3: 0.3842425308077874 betas[3,0]
   - 0.6942014022851634 betas[3,1] + 1.528221828396284 betas[3,2]
   + 0.7223183878646142 betas[3,3] + tau[3,3] = 0
 tau_computation_with_f_3_4: 0.3842425308077874 betas[4,0]
   - 0.6942014022851634 betas[4,1] + 1.528221828396284 betas[4,2]
   + 0.7223183878646142 betas[4,3] + tau[3,4] = 0
 tau_computation_with_f_3_5: 0.3842425308077874 betas[5,0]
   - 0.6942014022851634 betas[5,1] + 1.528221828396284 betas[5,2]
   + 0.7223183878646142 betas[5,3] + tau[3,5] = 0
 tau_computation_with_f_3_6: 0.3842425308077874 betas[6,0]
   - 0.6942014022851634 betas[6,1] + 1.528221828396284 betas[6,2]
   + 0.7223183878646142 betas[6,3] + tau[3,6] = 0
 tau_computation_with_f_3_7: 0.3842425308077874 betas[7,0]
   - 0.6942014022851634 betas[7,1] + 1.528221828396284 betas[7,2]
   + 0.7223183878646142 betas[7,3] + tau[3,7] = 0
 sum_r_3: r[3,0] + r[3,1] + r[3,2] + r[3,3] + r[3,4] + r[3,5] + r[3,6]
   + r[3,7] = 1
 Ensure_only_one_R_true_3: R[3,0] + R[3,1] + R[3,2] + R[3,3] + R[3,4]
   + R[3,5] + R[3,6] + R[3,7] = 1
 Define_Q_hat_3: - 953.6882234302366 R[3,0] - 623.7160437357866 R[3,1]
   - 679.4174930434373 R[3,2] - 714.7501185605312 R[3,3]
   - 742.1975949190615 R[3,4] - 742.1975469397994 R[3,5]
   - 755.6326435193275 R[3,6] - 763.3940180427094 R[3,7] + Q_hat[3] = 0
 Adjust_Q_hat_3: Q0_var[3] - Q_hat[3] + Q_hat_adjusted[3] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_3: Total_Demand_Up_to_K_minus_1[3]
   - 70.354593561725 R[3,0] - 135.8210602990659 R[3,1]
   - 182.4640142345818 R[3,2] - 247.5203819655339 R[3,3]
   - 247.5203819655339 R[3,4] - 312.5767498724398 R[3,5]
   - 503.436815918934 R[3,6] - 633.2812822562447 R[3,7] = 0
 Constr_Total_Demand_from_K_to_T_3: Total_Demand_from_k_to_T[3]
   - 693.0394244900285 R[3,0] - 627.5729577526877 R[3,1]
   - 580.9300038171718 R[3,2] - 515.8736360862197 R[3,3]
   - 515.8736360862197 R[3,4] - 450.8172681793138 R[3,5]
   - 259.9572021328196 R[3,6] - 130.1127357955089 R[3,7] = 0
 Constr_Left_0_diff_aux_3: Sold_0[3] - Q0_var[3] + Left_0_aux_3 = 0
 Constr_Lost_0_diff_aux_3: - Total_Demand_Up_to_K_minus_1[3] + Q0_var[3]
   + Lost_0_aux_3 = 0
 Constr_Q1_plus_left_3: - Left_0[3] + Q1_plus_left[3] - Q1_var[3] = 0
 Constr_Left_1_diff_aux_3: Sold_1[3] - Q1_plus_left[3] + Left_0_aux_3 = 0
 Constr_Lost_1_diff_aux_3: - Total_Demand_from_k_to_T[3] + Q1_plus_left[3]
   + Lost_0_aux_3 = 0
 Profit_Constraint_3: - 500 Sold_0[3] + 500 Lost_0[3] - 500 Sold_1[3]
   + 500 Left_1[3] + 500 Lost_1[3] + profits_vars[3] = 0
 R88: 1.041396362027678 alphas[0] + 1.60130056329161 alphas[1]
   - 1.273867487915398 alphas[2] + 0.5213967003607619 alphas[3] - alphas[4]
   + f_var[4] = 0
 Q0_upper_bound_4: - 1453.252517908159 Fraction_for_second_order_amount[4]
   + Q0_var[4] = 0
 tau_computation_with_f_4_0: 1.041396362027678 betas[0,0]
   + 1.60130056329161 betas[0,1] - 1.273867487915398 betas[0,2]
   + 0.5213967003607619 betas[0,3] + tau[4,0] = 0
 tau_computation_with_f_4_1: 1.041396362027678 betas[1,0]
   + 1.60130056329161 betas[1,1] - 1.273867487915398 betas[1,2]
   + 0.5213967003607619 betas[1,3] + tau[4,1] = 0
 tau_computation_with_f_4_2: 1.041396362027678 betas[2,0]
   + 1.60130056329161 betas[2,1] - 1.273867487915398 betas[2,2]
   + 0.5213967003607619 betas[2,3] + tau[4,2] = 0
 tau_computation_with_f_4_3: 1.041396362027678 betas[3,0]
   + 1.60130056329161 betas[3,1] - 1.273867487915398 betas[3,2]
   + 0.5213967003607619 betas[3,3] + tau[4,3] = 0
 tau_computation_with_f_4_4: 1.041396362027678 betas[4,0]
   + 1.60130056329161 betas[4,1] - 1.273867487915398 betas[4,2]
   + 0.5213967003607619 betas[4,3] + tau[4,4] = 0
 tau_computation_with_f_4_5: 1.041396362027678 betas[5,0]
   + 1.60130056329161 betas[5,1] - 1.273867487915398 betas[5,2]
   + 0.5213967003607619 betas[5,3] + tau[4,5] = 0
 tau_computation_with_f_4_6: 1.041396362027678 betas[6,0]
   + 1.60130056329161 betas[6,1] - 1.273867487915398 betas[6,2]
   + 0.5213967003607619 betas[6,3] + tau[4,6] = 0
 tau_computation_with_f_4_7: 1.041396362027678 betas[7,0]
   + 1.60130056329161 betas[7,1] - 1.273867487915398 betas[7,2]
   + 0.5213967003607619 betas[7,3] + tau[4,7] = 0
 sum_r_4: r[4,0] + r[4,1] + r[4,2] + r[4,3] + r[4,4] + r[4,5] + r[4,6]
   + r[4,7] = 1
 Ensure_only_one_R_true_4: R[4,0] + R[4,1] + R[4,2] + R[4,3] + R[4,4]
   + R[4,5] + R[4,6] + R[4,7] = 1
 Define_Q_hat_4: - 2126.498169732301 R[4,0] - 2188.270426996781 R[4,1]
   - 2548.608853379785 R[4,2] - 2547.420618751502 R[4,3]
   - 2655.035881567954 R[4,4] - 2655.035895127027 R[4,5]
   - 2644.656632841736 R[4,6] - 2632.799712339834 R[4,7] + Q_hat[4] = 0
 Adjust_Q_hat_4: Q0_var[4] - Q_hat[4] + Q_hat_adjusted[4] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_4: Total_Demand_Up_to_K_minus_1[4]
   - 369.0838016555306 R[4,0] - 585.7754819606682 R[4,1]
   - 1015.883422309999 R[4,2] - 1232.575101898519 R[4,3]
   - 1553.093401891735 R[4,4] - 1769.785080753152 R[4,5]
   - 1901.253602159537 R[4,6] - 2199.416354561477 R[4,7] = 0
 Constr_Total_Demand_from_K_to_T_4: Total_Demand_from_k_to_T[4]
   - 2263.71591062878 R[4,0] - 2047.024230323643 R[4,1]
   - 1616.916289974312 R[4,2] - 1400.224610385792 R[4,3]
   - 1079.706310392576 R[4,4] - 863.0146315311591 R[4,5]
   - 731.5461101247736 R[4,6] - 433.3833577228339 R[4,7] = 0
 Constr_Left_0_diff_aux_4: Sold_0[4] - Q0_var[4] + Left_0_aux_4 = 0
 Constr_Lost_0_diff_aux_4: - Total_Demand_Up_to_K_minus_1[4] + Q0_var[4]
   + Lost_0_aux_4 = 0
 Constr_Q1_plus_left_4: - Left_0[4] + Q1_plus_left[4] - Q1_var[4] = 0
 Constr_Left_1_diff_aux_4: Sold_1[4] - Q1_plus_left[4] + Left_0_aux_4 = 0
 Constr_Lost_1_diff_aux_4: - Total_Demand_from_k_to_T[4] + Q1_plus_left[4]
   + Lost_0_aux_4 = 0
 Profit_Constraint_4: - 500 Sold_0[4] + 500 Lost_0[4] - 500 Sold_1[4]
   + 500 Left_1[4] + 500 Lost_1[4] + profits_vars[4] = 0
 R110: - 0.1633166932572295 alphas[0] - 0.2068265727639536 alphas[1]
   - 1.29175138263669 alphas[2] - 0.0708562603421244 alphas[3] - alphas[4]
   + f_var[5] = 0
 Q0_upper_bound_5: - 1453.252517908159 Fraction_for_second_order_amount[5]
   + Q0_var[5] = 0
 tau_computation_with_f_5_0: - 0.1633166932572295 betas[0,0]
   - 0.2068265727639536 betas[0,1] - 1.29175138263669 betas[0,2]
   - 0.0708562603421244 betas[0,3] + tau[5,0] = 0
 tau_computation_with_f_5_1: - 0.1633166932572295 betas[1,0]
   - 0.2068265727639536 betas[1,1] - 1.29175138263669 betas[1,2]
   - 0.0708562603421244 betas[1,3] + tau[5,1] = 0
 tau_computation_with_f_5_2: - 0.1633166932572295 betas[2,0]
   - 0.2068265727639536 betas[2,1] - 1.29175138263669 betas[2,2]
   - 0.0708562603421244 betas[2,3] + tau[5,2] = 0
 tau_computation_with_f_5_3: - 0.1633166932572295 betas[3,0]
   - 0.2068265727639536 betas[3,1] - 1.29175138263669 betas[3,2]
   - 0.0708562603421244 betas[3,3] + tau[5,3] = 0
 tau_computation_with_f_5_4: - 0.1633166932572295 betas[4,0]
   - 0.2068265727639536 betas[4,1] - 1.29175138263669 betas[4,2]
   - 0.0708562603421244 betas[4,3] + tau[5,4] = 0
 tau_computation_with_f_5_5: - 0.1633166932572295 betas[5,0]
   - 0.2068265727639536 betas[5,1] - 1.29175138263669 betas[5,2]
   - 0.0708562603421244 betas[5,3] + tau[5,5] = 0
 tau_computation_with_f_5_6: - 0.1633166932572295 betas[6,0]
   - 0.2068265727639536 betas[6,1] - 1.29175138263669 betas[6,2]
   - 0.0708562603421244 betas[6,3] + tau[5,6] = 0
 tau_computation_with_f_5_7: - 0.1633166932572295 betas[7,0]
   - 0.2068265727639536 betas[7,1] - 1.29175138263669 betas[7,2]
   - 0.0708562603421244 betas[7,3] + tau[5,7] = 0
 sum_r_5: r[5,0] + r[5,1] + r[5,2] + r[5,3] + r[5,4] + r[5,5] + r[5,6]
   + r[5,7] = 1
 Ensure_only_one_R_true_5: R[5,0] + R[5,1] + R[5,2] + R[5,3] + R[5,4]
   + R[5,5] + R[5,6] + R[5,7] = 1
 Define_Q_hat_5: - 1855.381996025067 R[5,0] - 2153.522787712349 R[5,1]
   - 2228.900160553544 R[5,2] - 2237.265868048731 R[5,3]
   - 2109.523756728927 R[5,4] - 2109.523740290852 R[5,5]
   - 2177.243006814486 R[5,6] - 2191.952756302805 R[5,7] + Q_hat[5] = 0
 Adjust_Q_hat_5: Q0_var[5] - Q_hat[5] + Q_hat_adjusted[5] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_5: Total_Demand_Up_to_K_minus_1[5]
   - 300.0271568612467 R[5,0] - 517.6866273900624 R[5,1]
   - 794.8049593079409 R[5,2] - 1012.464426668792 R[5,3]
   - 1091.959317879675 R[5,4] - 1309.618785284215 R[5,5]
   - 1545.740087372096 R[5,6] - 1756.633821495953 R[5,7] = 0
 Constr_Total_Demand_from_K_to_T_5: Total_Demand_from_k_to_T[5]
   - 1891.925599443785 R[5,0] - 1674.26612891497 R[5,1]
   - 1397.147796997091 R[5,2] - 1179.48832963624 R[5,3]
   - 1099.993438425357 R[5,4] - 882.3339710208174 R[5,5]
   - 646.2126689329364 R[5,6] - 435.3189348090797 R[5,7] = 0
 Constr_Left_0_diff_aux_5: Sold_0[5] - Q0_var[5] + Left_0_aux_5 = 0
 Constr_Lost_0_diff_aux_5: - Total_Demand_Up_to_K_minus_1[5] + Q0_var[5]
   + Lost_0_aux_5 = 0
 Constr_Q1_plus_left_5: - Left_0[5] + Q1_plus_left[5] - Q1_var[5] = 0
 Constr_Left_1_diff_aux_5: Sold_1[5] - Q1_plus_left[5] + Left_0_aux_5 = 0
 Constr_Lost_1_diff_aux_5: - Total_Demand_from_k_to_T[5] + Q1_plus_left[5]
   + Lost_0_aux_5 = 0
 Profit_Constraint_5: - 500 Sold_0[5] + 500 Lost_0[5] - 500 Sold_1[5]
   + 500 Left_1[5] + 500 Lost_1[5] + profits_vars[5] = 0
 R132: 0.9658716573538862 alphas[0] + 1.514372419818931 alphas[1]
   + 0.8345285628671657 alphas[2] - 1.62687302729598 alphas[3] - alphas[4]
   + f_var[6] = 0
 Q0_upper_bound_6: - 1453.252517908159 Fraction_for_second_order_amount[6]
   + Q0_var[6] = 0
 tau_computation_with_f_6_0: 0.9658716573538862 betas[0,0]
   + 1.514372419818931 betas[0,1] + 0.8345285628671657 betas[0,2]
   - 1.62687302729598 betas[0,3] + tau[6,0] = 0
 tau_computation_with_f_6_1: 0.9658716573538862 betas[1,0]
   + 1.514372419818931 betas[1,1] + 0.8345285628671657 betas[1,2]
   - 1.62687302729598 betas[1,3] + tau[6,1] = 0
 tau_computation_with_f_6_2: 0.9658716573538862 betas[2,0]
   + 1.514372419818931 betas[2,1] + 0.8345285628671657 betas[2,2]
   - 1.62687302729598 betas[2,3] + tau[6,2] = 0
 tau_computation_with_f_6_3: 0.9658716573538862 betas[3,0]
   + 1.514372419818931 betas[3,1] + 0.8345285628671657 betas[3,2]
   - 1.62687302729598 betas[3,3] + tau[6,3] = 0
 tau_computation_with_f_6_4: 0.9658716573538862 betas[4,0]
   + 1.514372419818931 betas[4,1] + 0.8345285628671657 betas[4,2]
   - 1.62687302729598 betas[4,3] + tau[6,4] = 0
 tau_computation_with_f_6_5: 0.9658716573538862 betas[5,0]
   + 1.514372419818931 betas[5,1] + 0.8345285628671657 betas[5,2]
   - 1.62687302729598 betas[5,3] + tau[6,5] = 0
 tau_computation_with_f_6_6: 0.9658716573538862 betas[6,0]
   + 1.514372419818931 betas[6,1] + 0.8345285628671657 betas[6,2]
   - 1.62687302729598 betas[6,3] + tau[6,6] = 0
 tau_computation_with_f_6_7: 0.9658716573538862 betas[7,0]
   + 1.514372419818931 betas[7,1] + 0.8345285628671657 betas[7,2]
   - 1.62687302729598 betas[7,3] + tau[6,7] = 0
 sum_r_6: r[6,0] + r[6,1] + r[6,2] + r[6,3] + r[6,4] + r[6,5] + r[6,6]
   + r[6,7] = 1
 Ensure_only_one_R_true_6: R[6,0] + R[6,1] + R[6,2] + R[6,3] + R[6,4]
   + R[6,5] + R[6,6] + R[6,7] = 1
 Define_Q_hat_6: - 1304.661679831145 R[6,0] - 1018.043222230528 R[6,1]
   - 1173.443824624577 R[6,2] - 1151.100380931627 R[6,3]
   - 1101.452995660219 R[6,4] - 1101.452993149856 R[6,5]
   - 1123.440955855242 R[6,6] - 1147.995408466625 R[6,7] + Q_hat[6] = 0
 Adjust_Q_hat_6: Q0_var[6] - Q_hat[6] + Q_hat_adjusted[6] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_6: Total_Demand_Up_to_K_minus_1[6]
   - 159.7518765769098 R[6,0] - 262.3551653354041 R[6,1]
   - 425.9539816602642 R[6,2] - 528.5496214220292 R[6,3]
   - 528.5496214220292 R[6,4] - 631.1452616904894 R[6,5]
   - 749.972997430906 R[6,6] - 942.8041280121209 R[6,7] = 0
 Constr_Total_Demand_from_K_to_T_6: Total_Demand_from_k_to_T[6]
   - 988.2435319721316 R[6,0] - 885.6402432136374 R[6,1]
   - 722.0414268887772 R[6,2] - 619.4457871270122 R[6,3]
   - 619.4457871270122 R[6,4] - 516.850146858552 R[6,5]
   - 398.0224111181353 R[6,6] - 205.1912805369205 R[6,7] = 0
 Constr_Left_0_diff_aux_6: Sold_0[6] - Q0_var[6] + Left_0_aux_6 = 0
 Constr_Lost_0_diff_aux_6: - Total_Demand_Up_to_K_minus_1[6] + Q0_var[6]
   + Lost_0_aux_6 = 0
 Constr_Q1_plus_left_6: - Left_0[6] + Q1_plus_left[6] - Q1_var[6] = 0
 Constr_Left_1_diff_aux_6: Sold_1[6] - Q1_plus_left[6] + Left_0_aux_6 = 0
 Constr_Lost_1_diff_aux_6: - Total_Demand_from_k_to_T[6] + Q1_plus_left[6]
   + Lost_0_aux_6 = 0
 Profit_Constraint_6: - 500 Sold_0[6] + 500 Lost_0[6] - 500 Sold_1[6]
   + 500 Left_1[6] + 500 Lost_1[6] + profits_vars[6] = 0
 R154: - 1.496174957986087 alphas[0] - 1.263141220857213 alphas[1]
   + 1.210838876139294 alphas[2] + 0.8636588488600513 alphas[3] - alphas[4]
   + f_var[7] = 0
 Q0_upper_bound_7: - 1453.252517908159 Fraction_for_second_order_amount[7]
   + Q0_var[7] = 0
 tau_computation_with_f_7_0: - 1.496174957986087 betas[0,0]
   - 1.263141220857213 betas[0,1] + 1.210838876139294 betas[0,2]
   + 0.8636588488600513 betas[0,3] + tau[7,0] = 0
 tau_computation_with_f_7_1: - 1.496174957986087 betas[1,0]
   - 1.263141220857213 betas[1,1] + 1.210838876139294 betas[1,2]
   + 0.8636588488600513 betas[1,3] + tau[7,1] = 0
 tau_computation_with_f_7_2: - 1.496174957986087 betas[2,0]
   - 1.263141220857213 betas[2,1] + 1.210838876139294 betas[2,2]
   + 0.8636588488600513 betas[2,3] + tau[7,2] = 0
 tau_computation_with_f_7_3: - 1.496174957986087 betas[3,0]
   - 1.263141220857213 betas[3,1] + 1.210838876139294 betas[3,2]
   + 0.8636588488600513 betas[3,3] + tau[7,3] = 0
 tau_computation_with_f_7_4: - 1.496174957986087 betas[4,0]
   - 1.263141220857213 betas[4,1] + 1.210838876139294 betas[4,2]
   + 0.8636588488600513 betas[4,3] + tau[7,4] = 0
 tau_computation_with_f_7_5: - 1.496174957986087 betas[5,0]
   - 1.263141220857213 betas[5,1] + 1.210838876139294 betas[5,2]
   + 0.8636588488600513 betas[5,3] + tau[7,5] = 0
 tau_computation_with_f_7_6: - 1.496174957986087 betas[6,0]
   - 1.263141220857213 betas[6,1] + 1.210838876139294 betas[6,2]
   + 0.8636588488600513 betas[6,3] + tau[7,6] = 0
 tau_computation_with_f_7_7: - 1.496174957986087 betas[7,0]
   - 1.263141220857213 betas[7,1] + 1.210838876139294 betas[7,2]
   + 0.8636588488600513 betas[7,3] + tau[7,7] = 0
 sum_r_7: r[7,0] + r[7,1] + r[7,2] + r[7,3] + r[7,4] + r[7,5] + r[7,6]
   + r[7,7] = 1
 Ensure_only_one_R_true_7: R[7,0] + R[7,1] + R[7,2] + R[7,3] + R[7,4]
   + R[7,5] + R[7,6] + R[7,7] = 1
 Define_Q_hat_7: - 677.4763068796458 R[7,0] - 732.7157955946009 R[7,1]
   - 848.9242214611857 R[7,2] - 842.4778003286885 R[7,3]
   - 734.4315601442327 R[7,4] - 734.4315998839592 R[7,5]
   - 668.1559673772776 R[7,6] - 634.0713428049906 R[7,7] + Q_hat[7] = 0
 Adjust_Q_hat_7: Q0_var[7] - Q_hat[7] + Q_hat_adjusted[7] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_7: Total_Demand_Up_to_K_minus_1[7]
   - 82.32069700650572 R[7,1] - 82.32069700650572 R[7,2]
   - 164.5522711655415 R[7,3] - 199.9744554204937 R[7,4]
   - 282.206029834247 R[7,5] - 377.2802535349668 R[7,6]
   - 469.6081939327933 R[7,7] = 0
 Constr_Total_Demand_from_K_to_T_7: Total_Demand_from_k_to_T[7]
   - 634.0713427736039 R[7,0] - 551.7506457670979 R[7,1]
   - 551.750645767098 R[7,2] - 469.5190716080623 R[7,3]
   - 434.0968873531101 R[7,4] - 351.8653129393567 R[7,5]
   - 256.7910892386369 R[7,6] - 164.4631488408105 R[7,7] = 0
 Constr_Left_0_diff_aux_7: Sold_0[7] - Q0_var[7] + Left_0_aux_7 = 0
 Constr_Lost_0_diff_aux_7: - Total_Demand_Up_to_K_minus_1[7] + Q0_var[7]
   + Lost_0_aux_7 = 0
 Constr_Q1_plus_left_7: - Left_0[7] + Q1_plus_left[7] - Q1_var[7] = 0
 Constr_Left_1_diff_aux_7: Sold_1[7] - Q1_plus_left[7] + Left_0_aux_7 = 0
 Constr_Lost_1_diff_aux_7: - Total_Demand_from_k_to_T[7] + Q1_plus_left[7]
   + Lost_0_aux_7 = 0
 Profit_Constraint_7: - 500 Sold_0[7] + 500 Lost_0[7] - 500 Sold_1[7]
   + 500 Left_1[7] + 500 Lost_1[7] + profits_vars[7] = 0
 R176: - 1.885874379848093 alphas[0] + 0.202436046779539 alphas[1]
   + 0.3492397412676438 alphas[2] - 1.685194843566105 alphas[3] - alphas[4]
   + f_var[8] = 0
 Q0_upper_bound_8: - 1453.252517908159 Fraction_for_second_order_amount[8]
   + Q0_var[8] = 0
 tau_computation_with_f_8_0: - 1.885874379848093 betas[0,0]
   + 0.202436046779539 betas[0,1] + 0.3492397412676438 betas[0,2]
   - 1.685194843566105 betas[0,3] + tau[8,0] = 0
 tau_computation_with_f_8_1: - 1.885874379848093 betas[1,0]
   + 0.202436046779539 betas[1,1] + 0.3492397412676438 betas[1,2]
   - 1.685194843566105 betas[1,3] + tau[8,1] = 0
 tau_computation_with_f_8_2: - 1.885874379848093 betas[2,0]
   + 0.202436046779539 betas[2,1] + 0.3492397412676438 betas[2,2]
   - 1.685194843566105 betas[2,3] + tau[8,2] = 0
 tau_computation_with_f_8_3: - 1.885874379848093 betas[3,0]
   + 0.202436046779539 betas[3,1] + 0.3492397412676438 betas[3,2]
   - 1.685194843566105 betas[3,3] + tau[8,3] = 0
 tau_computation_with_f_8_4: - 1.885874379848093 betas[4,0]
   + 0.202436046779539 betas[4,1] + 0.3492397412676438 betas[4,2]
   - 1.685194843566105 betas[4,3] + tau[8,4] = 0
 tau_computation_with_f_8_5: - 1.885874379848093 betas[5,0]
   + 0.202436046779539 betas[5,1] + 0.3492397412676438 betas[5,2]
   - 1.685194843566105 betas[5,3] + tau[8,5] = 0
 tau_computation_with_f_8_6: - 1.885874379848093 betas[6,0]
   + 0.202436046779539 betas[6,1] + 0.3492397412676438 betas[6,2]
   - 1.685194843566105 betas[6,3] + tau[8,6] = 0
 tau_computation_with_f_8_7: - 1.885874379848093 betas[7,0]
   + 0.202436046779539 betas[7,1] + 0.3492397412676438 betas[7,2]
   - 1.685194843566105 betas[7,3] + tau[8,7] = 0
 sum_r_8: r[8,0] + r[8,1] + r[8,2] + r[8,3] + r[8,4] + r[8,5] + r[8,6]
   + r[8,7] = 1
 Ensure_only_one_R_true_8: R[8,0] + R[8,1] + R[8,2] + R[8,3] + R[8,4]
   + R[8,5] + R[8,6] + R[8,7] = 1
 Define_Q_hat_8: - 1546.800158206537 R[8,0] - 1295.826724741902 R[8,1]
   - 1107.592618675568 R[8,2] - 1095.212530435357 R[8,3]
   - 1099.665890404341 R[8,4] - 1099.665902222254 R[8,5]
   - 1100.63329262925 R[8,6] - 1128.601039437272 R[8,7] + Q_hat[8] = 0
 Adjust_Q_hat_8: Q0_var[8] - Q_hat[8] + Q_hat_adjusted[8] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_8: Total_Demand_Up_to_K_minus_1[8]
   - 221.4275437403683 R[8,0] - 350.2819547643519 R[8,1]
   - 467.2055459956493 R[8,2] - 596.0626333298927 R[8,3]
   - 596.0626333298927 R[8,4] - 724.9197210609386 R[8,5]
   - 810.1387961202072 R[8,6] - 870.8868638528354 R[8,7] = 0
 Constr_Total_Demand_from_K_to_T_8: Total_Demand_from_k_to_T[8]
   - 907.173495574559 R[8,0] - 778.3190845505753 R[8,1]
   - 661.3954933192779 R[8,2] - 532.5384059850345 R[8,3]
   - 532.5384059850345 R[8,4] - 403.6813182539885 R[8,5]
   - 318.4622431947199 R[8,6] - 257.7141754620919 R[8,7] = 0
 Constr_Left_0_diff_aux_8: Sold_0[8] - Q0_var[8] + Left_0_aux_8 = 0
 Constr_Lost_0_diff_aux_8: - Total_Demand_Up_to_K_minus_1[8] + Q0_var[8]
   + Lost_0_aux_8 = 0
 Constr_Q1_plus_left_8: - Left_0[8] + Q1_plus_left[8] - Q1_var[8] = 0
 Constr_Left_1_diff_aux_8: Sold_1[8] - Q1_plus_left[8] + Left_0_aux_8 = 0
 Constr_Lost_1_diff_aux_8: - Total_Demand_from_k_to_T[8] + Q1_plus_left[8]
   + Lost_0_aux_8 = 0
 Profit_Constraint_8: - 500 Sold_0[8] + 500 Lost_0[8] - 500 Sold_1[8]
   + 500 Left_1[8] + 500 Lost_1[8] + profits_vars[8] = 0
 R198: 1.259384184332354 alphas[0] + 0.5739646010964145 alphas[1]
   + 0.1261854070758312 alphas[2] + 0.867109190387246 alphas[3] - alphas[4]
   + f_var[9] = 0
 Q0_upper_bound_9: - 1453.252517908159 Fraction_for_second_order_amount[9]
   + Q0_var[9] = 0
 tau_computation_with_f_9_0: 1.259384184332354 betas[0,0]
   + 0.5739646010964145 betas[0,1] + 0.1261854070758312 betas[0,2]
   + 0.867109190387246 betas[0,3] + tau[9,0] = 0
 tau_computation_with_f_9_1: 1.259384184332354 betas[1,0]
   + 0.5739646010964145 betas[1,1] + 0.1261854070758312 betas[1,2]
   + 0.867109190387246 betas[1,3] + tau[9,1] = 0
 tau_computation_with_f_9_2: 1.259384184332354 betas[2,0]
   + 0.5739646010964145 betas[2,1] + 0.1261854070758312 betas[2,2]
   + 0.867109190387246 betas[2,3] + tau[9,2] = 0
 tau_computation_with_f_9_3: 1.259384184332354 betas[3,0]
   + 0.5739646010964145 betas[3,1] + 0.1261854070758312 betas[3,2]
   + 0.867109190387246 betas[3,3] + tau[9,3] = 0
 tau_computation_with_f_9_4: 1.259384184332354 betas[4,0]
   + 0.5739646010964145 betas[4,1] + 0.1261854070758312 betas[4,2]
   + 0.867109190387246 betas[4,3] + tau[9,4] = 0
 tau_computation_with_f_9_5: 1.259384184332354 betas[5,0]
   + 0.5739646010964145 betas[5,1] + 0.1261854070758312 betas[5,2]
   + 0.867109190387246 betas[5,3] + tau[9,5] = 0
 tau_computation_with_f_9_6: 1.259384184332354 betas[6,0]
   + 0.5739646010964145 betas[6,1] + 0.1261854070758312 betas[6,2]
   + 0.867109190387246 betas[6,3] + tau[9,6] = 0
 tau_computation_with_f_9_7: 1.259384184332354 betas[7,0]
   + 0.5739646010964145 betas[7,1] + 0.1261854070758312 betas[7,2]
   + 0.867109190387246 betas[7,3] + tau[9,7] = 0
 sum_r_9: r[9,0] + r[9,1] + r[9,2] + r[9,3] + r[9,4] + r[9,5] + r[9,6]
   + r[9,7] = 1
 Ensure_only_one_R_true_9: R[9,0] + R[9,1] + R[9,2] + R[9,3] + R[9,4]
   + R[9,5] + R[9,6] + R[9,7] = 1
 Define_Q_hat_9: - 1537.745616480408 R[9,0] - 1404.229418165119 R[9,1]
   - 1247.447469745689 R[9,2] - 1238.920486797099 R[9,3]
   - 1331.269284017422 R[9,4] - 1331.269315439335 R[9,5]
   - 1376.385265523305 R[9,6] - 1332.750318320855 R[9,7] + Q_hat[9] = 0
 Adjust_Q_hat_9: Q0_var[9] - Q_hat[9] + Q_hat_adjusted[9] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_9: Total_Demand_Up_to_K_minus_1[9]
   - 219.1212399031376 R[9,0] - 360.047930252907 R[9,1]
   - 485.1634403003203 R[9,2] - 626.0911330385218 R[9,3]
   - 769.1237613238598 R[9,4] - 910.0514540595154 R[9,5]
   - 1050.894932896207 R[9,6] - 1050.894932896207 R[9,7] = 0
 Constr_Total_Demand_from_K_to_T_9: Total_Demand_from_k_to_T[9]
   - 1113.62907846438 R[9,0] - 972.7023881146112 R[9,1]
   - 847.5868780671979 R[9,2] - 706.6591853289964 R[9,3]
   - 563.6265570436584 R[9,4] - 422.6988643080027 R[9,5]
   - 281.8553854713113 R[9,6] - 281.8553854713113 R[9,7] = 0
 Constr_Left_0_diff_aux_9: Sold_0[9] - Q0_var[9] + Left_0_aux_9 = 0
 Constr_Lost_0_diff_aux_9: - Total_Demand_Up_to_K_minus_1[9] + Q0_var[9]
   + Lost_0_aux_9 = 0
 Constr_Q1_plus_left_9: - Left_0[9] + Q1_plus_left[9] - Q1_var[9] = 0
 Constr_Left_1_diff_aux_9: Sold_1[9] - Q1_plus_left[9] + Left_0_aux_9 = 0
 Constr_Lost_1_diff_aux_9: - Total_Demand_from_k_to_T[9] + Q1_plus_left[9]
   + Lost_0_aux_9 = 0
 Profit_Constraint_9: - 500 Sold_0[9] + 500 Lost_0[9] - 500 Sold_1[9]
   + 500 Left_1[9] + 500 Lost_1[9] + profits_vars[9] = 0
 softmax_0_7: - exp_tau_var_0_7 + [ r[0,7] * exp_tau_var_0_0
   + r[0,7] * exp_tau_var_0_1 + r[0,7] * exp_tau_var_0_2
   + r[0,7] * exp_tau_var_0_3 + r[0,7] * exp_tau_var_0_4
   + r[0,7] * exp_tau_var_0_5 + r[0,7] * exp_tau_var_0_6
   + r[0,7] * exp_tau_var_0_7 ] = 0
 softmax_1_7: - exp_tau_var_1_7 + [ r[1,7] * exp_tau_var_1_0
   + r[1,7] * exp_tau_var_1_1 + r[1,7] * exp_tau_var_1_2
   + r[1,7] * exp_tau_var_1_3 + r[1,7] * exp_tau_var_1_4
   + r[1,7] * exp_tau_var_1_5 + r[1,7] * exp_tau_var_1_6
   + r[1,7] * exp_tau_var_1_7 ] = 0
 softmax_2_7: - exp_tau_var_2_7 + [ r[2,7] * exp_tau_var_2_0
   + r[2,7] * exp_tau_var_2_1 + r[2,7] * exp_tau_var_2_2
   + r[2,7] * exp_tau_var_2_3 + r[2,7] * exp_tau_var_2_4
   + r[2,7] * exp_tau_var_2_5 + r[2,7] * exp_tau_var_2_6
   + r[2,7] * exp_tau_var_2_7 ] = 0
 softmax_3_7: - exp_tau_var_3_7 + [ r[3,7] * exp_tau_var_3_0
   + r[3,7] * exp_tau_var_3_1 + r[3,7] * exp_tau_var_3_2
   + r[3,7] * exp_tau_var_3_3 + r[3,7] * exp_tau_var_3_4
   + r[3,7] * exp_tau_var_3_5 + r[3,7] * exp_tau_var_3_6
   + r[3,7] * exp_tau_var_3_7 ] = 0
 softmax_4_7: - exp_tau_var_4_7 + [ r[4,7] * exp_tau_var_4_0
   + r[4,7] * exp_tau_var_4_1 + r[4,7] * exp_tau_var_4_2
   + r[4,7] * exp_tau_var_4_3 + r[4,7] * exp_tau_var_4_4
   + r[4,7] * exp_tau_var_4_5 + r[4,7] * exp_tau_var_4_6
   + r[4,7] * exp_tau_var_4_7 ] = 0
 softmax_5_7: - exp_tau_var_5_7 + [ r[5,7] * exp_tau_var_5_0
   + r[5,7] * exp_tau_var_5_1 + r[5,7] * exp_tau_var_5_2
   + r[5,7] * exp_tau_var_5_3 + r[5,7] * exp_tau_var_5_4
   + r[5,7] * exp_tau_var_5_5 + r[5,7] * exp_tau_var_5_6
   + r[5,7] * exp_tau_var_5_7 ] = 0
 softmax_6_7: - exp_tau_var_6_7 + [ r[6,7] * exp_tau_var_6_0
   + r[6,7] * exp_tau_var_6_1 + r[6,7] * exp_tau_var_6_2
   + r[6,7] * exp_tau_var_6_3 + r[6,7] * exp_tau_var_6_4
   + r[6,7] * exp_tau_var_6_5 + r[6,7] * exp_tau_var_6_6
   + r[6,7] * exp_tau_var_6_7 ] = 0
 softmax_7_7: - exp_tau_var_7_7 + [ r[7,7] * exp_tau_var_7_0
   + r[7,7] * exp_tau_var_7_1 + r[7,7] * exp_tau_var_7_2
   + r[7,7] * exp_tau_var_7_3 + r[7,7] * exp_tau_var_7_4
   + r[7,7] * exp_tau_var_7_5 + r[7,7] * exp_tau_var_7_6
   + r[7,7] * exp_tau_var_7_7 ] = 0
 softmax_8_7: - exp_tau_var_8_7 + [ r[8,7] * exp_tau_var_8_0
   + r[8,7] * exp_tau_var_8_1 + r[8,7] * exp_tau_var_8_2
   + r[8,7] * exp_tau_var_8_3 + r[8,7] * exp_tau_var_8_4
   + r[8,7] * exp_tau_var_8_5 + r[8,7] * exp_tau_var_8_6
   + r[8,7] * exp_tau_var_8_7 ] = 0
 softmax_9_7: - exp_tau_var_9_7 + [ r[9,7] * exp_tau_var_9_0
   + r[9,7] * exp_tau_var_9_1 + r[9,7] * exp_tau_var_9_2
   + r[9,7] * exp_tau_var_9_3 + r[9,7] * exp_tau_var_9_4
   + r[9,7] * exp_tau_var_9_5 + r[9,7] * exp_tau_var_9_6
   + r[9,7] * exp_tau_var_9_7 ] = 0
 GC10: R[0,0] = 1 -> r[0,0] - max_r_helper = 0
 GC11: R[0,1] = 1 -> r[0,1] - max_r_helper = 0
 GC12: R[0,2] = 1 -> r[0,2] - max_r_helper = 0
 GC13: R[0,3] = 1 -> r[0,3] - max_r_helper = 0
 GC14: R[0,4] = 1 -> r[0,4] - max_r_helper = 0
 GC15: R[0,5] = 1 -> r[0,5] - max_r_helper = 0
 GC16: R[0,6] = 1 -> r[0,6] - max_r_helper = 0
 GC17: R[0,7] = 1 -> r[0,7] - max_r_helper = 0
 GC35: R[1,0] = 1 -> r[1,0] - max_r_helper = 0
 GC36: R[1,1] = 1 -> r[1,1] - max_r_helper = 0
 GC37: R[1,2] = 1 -> r[1,2] - max_r_helper = 0
 GC38: R[1,3] = 1 -> r[1,3] - max_r_helper = 0
 GC39: R[1,4] = 1 -> r[1,4] - max_r_helper = 0
 GC40: R[1,5] = 1 -> r[1,5] - max_r_helper = 0
 GC41: R[1,6] = 1 -> r[1,6] - max_r_helper = 0
 GC42: R[1,7] = 1 -> r[1,7] - max_r_helper = 0
 GC60: R[2,0] = 1 -> r[2,0] - max_r_helper = 0
 GC61: R[2,1] = 1 -> r[2,1] - max_r_helper = 0
 GC62: R[2,2] = 1 -> r[2,2] - max_r_helper = 0
 GC63: R[2,3] = 1 -> r[2,3] - max_r_helper = 0
 GC64: R[2,4] = 1 -> r[2,4] - max_r_helper = 0
 GC65: R[2,5] = 1 -> r[2,5] - max_r_helper = 0
 GC66: R[2,6] = 1 -> r[2,6] - max_r_helper = 0
 GC67: R[2,7] = 1 -> r[2,7] - max_r_helper = 0
 GC85: R[3,0] = 1 -> r[3,0] - max_r_helper = 0
 GC86: R[3,1] = 1 -> r[3,1] - max_r_helper = 0
 GC87: R[3,2] = 1 -> r[3,2] - max_r_helper = 0
 GC88: R[3,3] = 1 -> r[3,3] - max_r_helper = 0
 GC89: R[3,4] = 1 -> r[3,4] - max_r_helper = 0
 GC90: R[3,5] = 1 -> r[3,5] - max_r_helper = 0
 GC91: R[3,6] = 1 -> r[3,6] - max_r_helper = 0
 GC92: R[3,7] = 1 -> r[3,7] - max_r_helper = 0
 GC110: R[4,0] = 1 -> r[4,0] - max_r_helper = 0
 GC111: R[4,1] = 1 -> r[4,1] - max_r_helper = 0
 GC112: R[4,2] = 1 -> r[4,2] - max_r_helper = 0
 GC113: R[4,3] = 1 -> r[4,3] - max_r_helper = 0
 GC114: R[4,4] = 1 -> r[4,4] - max_r_helper = 0
 GC115: R[4,5] = 1 -> r[4,5] - max_r_helper = 0
 GC116: R[4,6] = 1 -> r[4,6] - max_r_helper = 0
 GC117: R[4,7] = 1 -> r[4,7] - max_r_helper = 0
 GC135: R[5,0] = 1 -> r[5,0] - max_r_helper = 0
 GC136: R[5,1] = 1 -> r[5,1] - max_r_helper = 0
 GC137: R[5,2] = 1 -> r[5,2] - max_r_helper = 0
 GC138: R[5,3] = 1 -> r[5,3] - max_r_helper = 0
 GC139: R[5,4] = 1 -> r[5,4] - max_r_helper = 0
 GC140: R[5,5] = 1 -> r[5,5] - max_r_helper = 0
 GC141: R[5,6] = 1 -> r[5,6] - max_r_helper = 0
 GC142: R[5,7] = 1 -> r[5,7] - max_r_helper = 0
 GC160: R[6,0] = 1 -> r[6,0] - max_r_helper = 0
 GC161: R[6,1] = 1 -> r[6,1] - max_r_helper = 0
 GC162: R[6,2] = 1 -> r[6,2] - max_r_helper = 0
 GC163: R[6,3] = 1 -> r[6,3] - max_r_helper = 0
 GC164: R[6,4] = 1 -> r[6,4] - max_r_helper = 0
 GC165: R[6,5] = 1 -> r[6,5] - max_r_helper = 0
 GC166: R[6,6] = 1 -> r[6,6] - max_r_helper = 0
 GC167: R[6,7] = 1 -> r[6,7] - max_r_helper = 0
 GC185: R[7,0] = 1 -> r[7,0] - max_r_helper = 0
 GC186: R[7,1] = 1 -> r[7,1] - max_r_helper = 0
 GC187: R[7,2] = 1 -> r[7,2] - max_r_helper = 0
 GC188: R[7,3] = 1 -> r[7,3] - max_r_helper = 0
 GC189: R[7,4] = 1 -> r[7,4] - max_r_helper = 0
 GC190: R[7,5] = 1 -> r[7,5] - max_r_helper = 0
 GC191: R[7,6] = 1 -> r[7,6] - max_r_helper = 0
 GC192: R[7,7] = 1 -> r[7,7] - max_r_helper = 0
 GC210: R[8,0] = 1 -> r[8,0] - max_r_helper = 0
 GC211: R[8,1] = 1 -> r[8,1] - max_r_helper = 0
 GC212: R[8,2] = 1 -> r[8,2] - max_r_helper = 0
 GC213: R[8,3] = 1 -> r[8,3] - max_r_helper = 0
 GC214: R[8,4] = 1 -> r[8,4] - max_r_helper = 0
 GC215: R[8,5] = 1 -> r[8,5] - max_r_helper = 0
 GC216: R[8,6] = 1 -> r[8,6] - max_r_helper = 0
 GC217: R[8,7] = 1 -> r[8,7] - max_r_helper = 0
 GC235: R[9,0] = 1 -> r[9,0] - max_r_helper = 0
 GC236: R[9,1] = 1 -> r[9,1] - max_r_helper = 0
 GC237: R[9,2] = 1 -> r[9,2] - max_r_helper = 0
 GC238: R[9,3] = 1 -> r[9,3] - max_r_helper = 0
 GC239: R[9,4] = 1 -> r[9,4] - max_r_helper = 0
 GC240: R[9,5] = 1 -> r[9,5] - max_r_helper = 0
 GC241: R[9,6] = 1 -> r[9,6] - max_r_helper = 0
 GC242: R[9,7] = 1 -> r[9,7] - max_r_helper = 0
Bounds
 alphas[0] free
 alphas[1] free
 alphas[2] free
 alphas[3] free
 alphas[4] free
 betas[0,0] free
 betas[0,1] free
 betas[0,2] free
 betas[0,3] free
 betas[1,0] free
 betas[1,1] free
 betas[1,2] free
 betas[1,3] free
 betas[2,0] free
 betas[2,1] free
 betas[2,2] free
 betas[2,3] free
 betas[3,0] free
 betas[3,1] free
 betas[3,2] free
 betas[3,3] free
 betas[4,0] free
 betas[4,1] free
 betas[4,2] free
 betas[4,3] free
 betas[5,0] free
 betas[5,1] free
 betas[5,2] free
 betas[5,3] free
 betas[6,0] free
 betas[6,1] free
 betas[6,2] free
 betas[6,3] free
 betas[7,0] free
 betas[7,1] free
 betas[7,2] free
 betas[7,3] free
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
 Fraction_for_second_order_amount[0] <= 1
 Fraction_for_second_order_amount[1] <= 1
 Fraction_for_second_order_amount[2] <= 1
 Fraction_for_second_order_amount[3] <= 1
 Fraction_for_second_order_amount[4] <= 1
 Fraction_for_second_order_amount[5] <= 1
 Fraction_for_second_order_amount[6] <= 1
 Fraction_for_second_order_amount[7] <= 1
 Fraction_for_second_order_amount[8] <= 1
 Fraction_for_second_order_amount[9] <= 1
 Q0_var[0] <= 1454.252517908159
 Q0_var[1] <= 1454.252517908159
 Q0_var[2] <= 1454.252517908159
 Q0_var[3] <= 1454.252517908159
 Q0_var[4] <= 1454.252517908159
 Q0_var[5] <= 1454.252517908159
 Q0_var[6] <= 1454.252517908159
 Q0_var[7] <= 1454.252517908159
 Q0_var[8] <= 1454.252517908159
 Q0_var[9] <= 1454.252517908159
 tau[0,0] free
 tau[0,1] free
 tau[0,2] free
 tau[0,3] free
 tau[0,4] free
 tau[0,5] free
 tau[0,6] free
 tau[0,7] free
 tau[1,0] free
 tau[1,1] free
 tau[1,2] free
 tau[1,3] free
 tau[1,4] free
 tau[1,5] free
 tau[1,6] free
 tau[1,7] free
 tau[2,0] free
 tau[2,1] free
 tau[2,2] free
 tau[2,3] free
 tau[2,4] free
 tau[2,5] free
 tau[2,6] free
 tau[2,7] free
 tau[3,0] free
 tau[3,1] free
 tau[3,2] free
 tau[3,3] free
 tau[3,4] free
 tau[3,5] free
 tau[3,6] free
 tau[3,7] free
 tau[4,0] free
 tau[4,1] free
 tau[4,2] free
 tau[4,3] free
 tau[4,4] free
 tau[4,5] free
 tau[4,6] free
 tau[4,7] free
 tau[5,0] free
 tau[5,1] free
 tau[5,2] free
 tau[5,3] free
 tau[5,4] free
 tau[5,5] free
 tau[5,6] free
 tau[5,7] free
 tau[6,0] free
 tau[6,1] free
 tau[6,2] free
 tau[6,3] free
 tau[6,4] free
 tau[6,5] free
 tau[6,6] free
 tau[6,7] free
 tau[7,0] free
 tau[7,1] free
 tau[7,2] free
 tau[7,3] free
 tau[7,4] free
 tau[7,5] free
 tau[7,6] free
 tau[7,7] free
 tau[8,0] free
 tau[8,1] free
 tau[8,2] free
 tau[8,3] free
 tau[8,4] free
 tau[8,5] free
 tau[8,6] free
 tau[8,7] free
 tau[9,0] free
 tau[9,1] free
 tau[9,2] free
 tau[9,3] free
 tau[9,4] free
 tau[9,5] free
 tau[9,6] free
 tau[9,7] free
 r[0,0] <= 1
 r[0,1] <= 1
 r[0,2] <= 1
 r[0,3] <= 1
 r[0,4] <= 1
 r[0,5] <= 1
 r[0,6] <= 1
 r[0,7] <= 1
 r[1,0] <= 1
 r[1,1] <= 1
 r[1,2] <= 1
 r[1,3] <= 1
 r[1,4] <= 1
 r[1,5] <= 1
 r[1,6] <= 1
 r[1,7] <= 1
 r[2,0] <= 1
 r[2,1] <= 1
 r[2,2] <= 1
 r[2,3] <= 1
 r[2,4] <= 1
 r[2,5] <= 1
 r[2,6] <= 1
 r[2,7] <= 1
 r[3,0] <= 1
 r[3,1] <= 1
 r[3,2] <= 1
 r[3,3] <= 1
 r[3,4] <= 1
 r[3,5] <= 1
 r[3,6] <= 1
 r[3,7] <= 1
 r[4,0] <= 1
 r[4,1] <= 1
 r[4,2] <= 1
 r[4,3] <= 1
 r[4,4] <= 1
 r[4,5] <= 1
 r[4,6] <= 1
 r[4,7] <= 1
 r[5,0] <= 1
 r[5,1] <= 1
 r[5,2] <= 1
 r[5,3] <= 1
 r[5,4] <= 1
 r[5,5] <= 1
 r[5,6] <= 1
 r[5,7] <= 1
 r[6,0] <= 1
 r[6,1] <= 1
 r[6,2] <= 1
 r[6,3] <= 1
 r[6,4] <= 1
 r[6,5] <= 1
 r[6,6] <= 1
 r[6,7] <= 1
 r[7,0] <= 1
 r[7,1] <= 1
 r[7,2] <= 1
 r[7,3] <= 1
 r[7,4] <= 1
 r[7,5] <= 1
 r[7,6] <= 1
 r[7,7] <= 1
 r[8,0] <= 1
 r[8,1] <= 1
 r[8,2] <= 1
 r[8,3] <= 1
 r[8,4] <= 1
 r[8,5] <= 1
 r[8,6] <= 1
 r[8,7] <= 1
 r[9,0] <= 1
 r[9,1] <= 1
 r[9,2] <= 1
 r[9,3] <= 1
 r[9,4] <= 1
 r[9,5] <= 1
 r[9,6] <= 1
 r[9,7] <= 1
 Q_hat_adjusted[0] free
 Q_hat_adjusted[1] free
 Q_hat_adjusted[2] free
 Q_hat_adjusted[3] free
 Q_hat_adjusted[4] free
 Q_hat_adjusted[5] free
 Q_hat_adjusted[6] free
 Q_hat_adjusted[7] free
 Q_hat_adjusted[8] free
 Q_hat_adjusted[9] free
 max_r_helper <= 1
 Left_0_aux_0 free
 Lost_0_aux_0 free
 Left_0_aux_0 free
 Lost_0_aux_0 free
 max_r_helper <= 1
 Left_0_aux_1 free
 Lost_0_aux_1 free
 Left_0_aux_1 free
 Lost_0_aux_1 free
 max_r_helper <= 1
 Left_0_aux_2 free
 Lost_0_aux_2 free
 Left_0_aux_2 free
 Lost_0_aux_2 free
 max_r_helper <= 1
 Left_0_aux_3 free
 Lost_0_aux_3 free
 Left_0_aux_3 free
 Lost_0_aux_3 free
 max_r_helper <= 1
 Left_0_aux_4 free
 Lost_0_aux_4 free
 Left_0_aux_4 free
 Lost_0_aux_4 free
 max_r_helper <= 1
 Left_0_aux_5 free
 Lost_0_aux_5 free
 Left_0_aux_5 free
 Lost_0_aux_5 free
 max_r_helper <= 1
 Left_0_aux_6 free
 Lost_0_aux_6 free
 Left_0_aux_6 free
 Lost_0_aux_6 free
 max_r_helper <= 1
 Left_0_aux_7 free
 Lost_0_aux_7 free
 Left_0_aux_7 free
 Lost_0_aux_7 free
 max_r_helper <= 1
 Left_0_aux_8 free
 Lost_0_aux_8 free
 Left_0_aux_8 free
 Lost_0_aux_8 free
 max_r_helper <= 1
 Left_0_aux_9 free
 Lost_0_aux_9 free
 Left_0_aux_9 free
 Lost_0_aux_9 free
Binaries
 R[0,0] R[0,1] R[0,2] R[0,3] R[0,4] R[0,5] R[0,6] R[0,7] R[1,0] R[1,1]
 R[1,2] R[1,3] R[1,4] R[1,5] R[1,6] R[1,7] R[2,0] R[2,1] R[2,2] R[2,3]
 R[2,4] R[2,5] R[2,6] R[2,7] R[3,0] R[3,1] R[3,2] R[3,3] R[3,4] R[3,5]
 R[3,6] R[3,7] R[4,0] R[4,1] R[4,2] R[4,3] R[4,4] R[4,5] R[4,6] R[4,7]
 R[5,0] R[5,1] R[5,2] R[5,3] R[5,4] R[5,5] R[5,6] R[5,7] R[6,0] R[6,1]
 R[6,2] R[6,3] R[6,4] R[6,5] R[6,6] R[6,7] R[7,0] R[7,1] R[7,2] R[7,3]
 R[7,4] R[7,5] R[7,6] R[7,7] R[8,0] R[8,1] R[8,2] R[8,3] R[8,4] R[8,5]
 R[8,6] R[8,7] R[9,0] R[9,1] R[9,2] R[9,3] R[9,4] R[9,5] R[9,6] R[9,7]
General Constraints
 logistic_constraint_0: Fraction_for_second_order_amount[0] =
   LOGISTIC ( f_var[0] )
 GC1: exp_tau_var_0_0 = EXP ( tau[0,0] )
 GC2: exp_tau_var_0_1 = EXP ( tau[0,1] )
 GC3: exp_tau_var_0_2 = EXP ( tau[0,2] )
 GC4: exp_tau_var_0_3 = EXP ( tau[0,3] )
 GC5: exp_tau_var_0_4 = EXP ( tau[0,4] )
 GC6: exp_tau_var_0_5 = EXP ( tau[0,5] )
 GC7: exp_tau_var_0_6 = EXP ( tau[0,6] )
 GC8: exp_tau_var_0_7 = EXP ( tau[0,7] )
 MaxRConstraint_0: max_r_helper = MAX ( r[0,0] , r[0,1] , r[0,2] , r[0,3] ,
   r[0,4] , r[0,5] , r[0,6] , r[0,7] )
 Max_Constraint_0: Q1_var[0] = MAX ( Q_hat_adjusted[0] , 0 )
 Constr_Sold_0_min_0: Sold_0[0] = MIN ( Total_Demand_Up_to_K_minus_1[0] ,
   Q0_var[0] )
 Constr_Left_0_max_0: Left_0[0] = MAX ( Left_0_aux_0 , 0 )
 Constr_Lost_0_max_0: Lost_0[0] = MAX ( Lost_0_aux_0 , 0 )
 Constr_Sold_1_min_0: Sold_1[0] = MIN ( Total_Demand_from_k_to_T[0] ,
   Q1_plus_left[0] )
 Constr_Left_1_max_0: Left_1[0] = MAX ( Left_0_aux_0 , 0 )
 Constr_Lost_1_max_0: Lost_1[0] = MAX ( Lost_0_aux_0 , 0 )
 logistic_constraint_1: Fraction_for_second_order_amount[1] =
   LOGISTIC ( f_var[1] )
 GC26: exp_tau_var_1_0 = EXP ( tau[1,0] )
 GC27: exp_tau_var_1_1 = EXP ( tau[1,1] )
 GC28: exp_tau_var_1_2 = EXP ( tau[1,2] )
 GC29: exp_tau_var_1_3 = EXP ( tau[1,3] )
 GC30: exp_tau_var_1_4 = EXP ( tau[1,4] )
 GC31: exp_tau_var_1_5 = EXP ( tau[1,5] )
 GC32: exp_tau_var_1_6 = EXP ( tau[1,6] )
 GC33: exp_tau_var_1_7 = EXP ( tau[1,7] )
 MaxRConstraint_1: max_r_helper = MAX ( r[1,0] , r[1,1] , r[1,2] , r[1,3] ,
   r[1,4] , r[1,5] , r[1,6] , r[1,7] )
 Max_Constraint_1: Q1_var[1] = MAX ( Q_hat_adjusted[1] , 0 )
 Constr_Sold_0_min_1: Sold_0[1] = MIN ( Total_Demand_Up_to_K_minus_1[1] ,
   Q0_var[1] )
 Constr_Left_0_max_1: Left_0[1] = MAX ( Left_0_aux_1 , 0 )
 Constr_Lost_0_max_1: Lost_0[1] = MAX ( Lost_0_aux_1 , 0 )
 Constr_Sold_1_min_1: Sold_1[1] = MIN ( Total_Demand_from_k_to_T[1] ,
   Q1_plus_left[1] )
 Constr_Left_1_max_1: Left_1[1] = MAX ( Left_0_aux_1 , 0 )
 Constr_Lost_1_max_1: Lost_1[1] = MAX ( Lost_0_aux_1 , 0 )
 logistic_constraint_2: Fraction_for_second_order_amount[2] =
   LOGISTIC ( f_var[2] )
 GC51: exp_tau_var_2_0 = EXP ( tau[2,0] )
 GC52: exp_tau_var_2_1 = EXP ( tau[2,1] )
 GC53: exp_tau_var_2_2 = EXP ( tau[2,2] )
 GC54: exp_tau_var_2_3 = EXP ( tau[2,3] )
 GC55: exp_tau_var_2_4 = EXP ( tau[2,4] )
 GC56: exp_tau_var_2_5 = EXP ( tau[2,5] )
 GC57: exp_tau_var_2_6 = EXP ( tau[2,6] )
 GC58: exp_tau_var_2_7 = EXP ( tau[2,7] )
 MaxRConstraint_2: max_r_helper = MAX ( r[2,0] , r[2,1] , r[2,2] , r[2,3] ,
   r[2,4] , r[2,5] , r[2,6] , r[2,7] )
 Max_Constraint_2: Q1_var[2] = MAX ( Q_hat_adjusted[2] , 0 )
 Constr_Sold_0_min_2: Sold_0[2] = MIN ( Total_Demand_Up_to_K_minus_1[2] ,
   Q0_var[2] )
 Constr_Left_0_max_2: Left_0[2] = MAX ( Left_0_aux_2 , 0 )
 Constr_Lost_0_max_2: Lost_0[2] = MAX ( Lost_0_aux_2 , 0 )
 Constr_Sold_1_min_2: Sold_1[2] = MIN ( Total_Demand_from_k_to_T[2] ,
   Q1_plus_left[2] )
 Constr_Left_1_max_2: Left_1[2] = MAX ( Left_0_aux_2 , 0 )
 Constr_Lost_1_max_2: Lost_1[2] = MAX ( Lost_0_aux_2 , 0 )
 logistic_constraint_3: Fraction_for_second_order_amount[3] =
   LOGISTIC ( f_var[3] )
 GC76: exp_tau_var_3_0 = EXP ( tau[3,0] )
 GC77: exp_tau_var_3_1 = EXP ( tau[3,1] )
 GC78: exp_tau_var_3_2 = EXP ( tau[3,2] )
 GC79: exp_tau_var_3_3 = EXP ( tau[3,3] )
 GC80: exp_tau_var_3_4 = EXP ( tau[3,4] )
 GC81: exp_tau_var_3_5 = EXP ( tau[3,5] )
 GC82: exp_tau_var_3_6 = EXP ( tau[3,6] )
 GC83: exp_tau_var_3_7 = EXP ( tau[3,7] )
 MaxRConstraint_3: max_r_helper = MAX ( r[3,0] , r[3,1] , r[3,2] , r[3,3] ,
   r[3,4] , r[3,5] , r[3,6] , r[3,7] )
 Max_Constraint_3: Q1_var[3] = MAX ( Q_hat_adjusted[3] , 0 )
 Constr_Sold_0_min_3: Sold_0[3] = MIN ( Total_Demand_Up_to_K_minus_1[3] ,
   Q0_var[3] )
 Constr_Left_0_max_3: Left_0[3] = MAX ( Left_0_aux_3 , 0 )
 Constr_Lost_0_max_3: Lost_0[3] = MAX ( Lost_0_aux_3 , 0 )
 Constr_Sold_1_min_3: Sold_1[3] = MIN ( Total_Demand_from_k_to_T[3] ,
   Q1_plus_left[3] )
 Constr_Left_1_max_3: Left_1[3] = MAX ( Left_0_aux_3 , 0 )
 Constr_Lost_1_max_3: Lost_1[3] = MAX ( Lost_0_aux_3 , 0 )
 logistic_constraint_4: Fraction_for_second_order_amount[4] =
   LOGISTIC ( f_var[4] )
 GC101: exp_tau_var_4_0 = EXP ( tau[4,0] )
 GC102: exp_tau_var_4_1 = EXP ( tau[4,1] )
 GC103: exp_tau_var_4_2 = EXP ( tau[4,2] )
 GC104: exp_tau_var_4_3 = EXP ( tau[4,3] )
 GC105: exp_tau_var_4_4 = EXP ( tau[4,4] )
 GC106: exp_tau_var_4_5 = EXP ( tau[4,5] )
 GC107: exp_tau_var_4_6 = EXP ( tau[4,6] )
 GC108: exp_tau_var_4_7 = EXP ( tau[4,7] )
 MaxRConstraint_4: max_r_helper = MAX ( r[4,0] , r[4,1] , r[4,2] , r[4,3] ,
   r[4,4] , r[4,5] , r[4,6] , r[4,7] )
 Max_Constraint_4: Q1_var[4] = MAX ( Q_hat_adjusted[4] , 0 )
 Constr_Sold_0_min_4: Sold_0[4] = MIN ( Total_Demand_Up_to_K_minus_1[4] ,
   Q0_var[4] )
 Constr_Left_0_max_4: Left_0[4] = MAX ( Left_0_aux_4 , 0 )
 Constr_Lost_0_max_4: Lost_0[4] = MAX ( Lost_0_aux_4 , 0 )
 Constr_Sold_1_min_4: Sold_1[4] = MIN ( Total_Demand_from_k_to_T[4] ,
   Q1_plus_left[4] )
 Constr_Left_1_max_4: Left_1[4] = MAX ( Left_0_aux_4 , 0 )
 Constr_Lost_1_max_4: Lost_1[4] = MAX ( Lost_0_aux_4 , 0 )
 logistic_constraint_5: Fraction_for_second_order_amount[5] =
   LOGISTIC ( f_var[5] )
 GC126: exp_tau_var_5_0 = EXP ( tau[5,0] )
 GC127: exp_tau_var_5_1 = EXP ( tau[5,1] )
 GC128: exp_tau_var_5_2 = EXP ( tau[5,2] )
 GC129: exp_tau_var_5_3 = EXP ( tau[5,3] )
 GC130: exp_tau_var_5_4 = EXP ( tau[5,4] )
 GC131: exp_tau_var_5_5 = EXP ( tau[5,5] )
 GC132: exp_tau_var_5_6 = EXP ( tau[5,6] )
 GC133: exp_tau_var_5_7 = EXP ( tau[5,7] )
 MaxRConstraint_5: max_r_helper = MAX ( r[5,0] , r[5,1] , r[5,2] , r[5,3] ,
   r[5,4] , r[5,5] , r[5,6] , r[5,7] )
 Max_Constraint_5: Q1_var[5] = MAX ( Q_hat_adjusted[5] , 0 )
 Constr_Sold_0_min_5: Sold_0[5] = MIN ( Total_Demand_Up_to_K_minus_1[5] ,
   Q0_var[5] )
 Constr_Left_0_max_5: Left_0[5] = MAX ( Left_0_aux_5 , 0 )
 Constr_Lost_0_max_5: Lost_0[5] = MAX ( Lost_0_aux_5 , 0 )
 Constr_Sold_1_min_5: Sold_1[5] = MIN ( Total_Demand_from_k_to_T[5] ,
   Q1_plus_left[5] )
 Constr_Left_1_max_5: Left_1[5] = MAX ( Left_0_aux_5 , 0 )
 Constr_Lost_1_max_5: Lost_1[5] = MAX ( Lost_0_aux_5 , 0 )
 logistic_constraint_6: Fraction_for_second_order_amount[6] =
   LOGISTIC ( f_var[6] )
 GC151: exp_tau_var_6_0 = EXP ( tau[6,0] )
 GC152: exp_tau_var_6_1 = EXP ( tau[6,1] )
 GC153: exp_tau_var_6_2 = EXP ( tau[6,2] )
 GC154: exp_tau_var_6_3 = EXP ( tau[6,3] )
 GC155: exp_tau_var_6_4 = EXP ( tau[6,4] )
 GC156: exp_tau_var_6_5 = EXP ( tau[6,5] )
 GC157: exp_tau_var_6_6 = EXP ( tau[6,6] )
 GC158: exp_tau_var_6_7 = EXP ( tau[6,7] )
 MaxRConstraint_6: max_r_helper = MAX ( r[6,0] , r[6,1] , r[6,2] , r[6,3] ,
   r[6,4] , r[6,5] , r[6,6] , r[6,7] )
 Max_Constraint_6: Q1_var[6] = MAX ( Q_hat_adjusted[6] , 0 )
 Constr_Sold_0_min_6: Sold_0[6] = MIN ( Total_Demand_Up_to_K_minus_1[6] ,
   Q0_var[6] )
 Constr_Left_0_max_6: Left_0[6] = MAX ( Left_0_aux_6 , 0 )
 Constr_Lost_0_max_6: Lost_0[6] = MAX ( Lost_0_aux_6 , 0 )
 Constr_Sold_1_min_6: Sold_1[6] = MIN ( Total_Demand_from_k_to_T[6] ,
   Q1_plus_left[6] )
 Constr_Left_1_max_6: Left_1[6] = MAX ( Left_0_aux_6 , 0 )
 Constr_Lost_1_max_6: Lost_1[6] = MAX ( Lost_0_aux_6 , 0 )
 logistic_constraint_7: Fraction_for_second_order_amount[7] =
   LOGISTIC ( f_var[7] )
 GC176: exp_tau_var_7_0 = EXP ( tau[7,0] )
 GC177: exp_tau_var_7_1 = EXP ( tau[7,1] )
 GC178: exp_tau_var_7_2 = EXP ( tau[7,2] )
 GC179: exp_tau_var_7_3 = EXP ( tau[7,3] )
 GC180: exp_tau_var_7_4 = EXP ( tau[7,4] )
 GC181: exp_tau_var_7_5 = EXP ( tau[7,5] )
 GC182: exp_tau_var_7_6 = EXP ( tau[7,6] )
 GC183: exp_tau_var_7_7 = EXP ( tau[7,7] )
 MaxRConstraint_7: max_r_helper = MAX ( r[7,0] , r[7,1] , r[7,2] , r[7,3] ,
   r[7,4] , r[7,5] , r[7,6] , r[7,7] )
 Max_Constraint_7: Q1_var[7] = MAX ( Q_hat_adjusted[7] , 0 )
 Constr_Sold_0_min_7: Sold_0[7] = MIN ( Total_Demand_Up_to_K_minus_1[7] ,
   Q0_var[7] )
 Constr_Left_0_max_7: Left_0[7] = MAX ( Left_0_aux_7 , 0 )
 Constr_Lost_0_max_7: Lost_0[7] = MAX ( Lost_0_aux_7 , 0 )
 Constr_Sold_1_min_7: Sold_1[7] = MIN ( Total_Demand_from_k_to_T[7] ,
   Q1_plus_left[7] )
 Constr_Left_1_max_7: Left_1[7] = MAX ( Left_0_aux_7 , 0 )
 Constr_Lost_1_max_7: Lost_1[7] = MAX ( Lost_0_aux_7 , 0 )
 logistic_constraint_8: Fraction_for_second_order_amount[8] =
   LOGISTIC ( f_var[8] )
 GC201: exp_tau_var_8_0 = EXP ( tau[8,0] )
 GC202: exp_tau_var_8_1 = EXP ( tau[8,1] )
 GC203: exp_tau_var_8_2 = EXP ( tau[8,2] )
 GC204: exp_tau_var_8_3 = EXP ( tau[8,3] )
 GC205: exp_tau_var_8_4 = EXP ( tau[8,4] )
 GC206: exp_tau_var_8_5 = EXP ( tau[8,5] )
 GC207: exp_tau_var_8_6 = EXP ( tau[8,6] )
 GC208: exp_tau_var_8_7 = EXP ( tau[8,7] )
 MaxRConstraint_8: max_r_helper = MAX ( r[8,0] , r[8,1] , r[8,2] , r[8,3] ,
   r[8,4] , r[8,5] , r[8,6] , r[8,7] )
 Max_Constraint_8: Q1_var[8] = MAX ( Q_hat_adjusted[8] , 0 )
 Constr_Sold_0_min_8: Sold_0[8] = MIN ( Total_Demand_Up_to_K_minus_1[8] ,
   Q0_var[8] )
 Constr_Left_0_max_8: Left_0[8] = MAX ( Left_0_aux_8 , 0 )
 Constr_Lost_0_max_8: Lost_0[8] = MAX ( Lost_0_aux_8 , 0 )
 Constr_Sold_1_min_8: Sold_1[8] = MIN ( Total_Demand_from_k_to_T[8] ,
   Q1_plus_left[8] )
 Constr_Left_1_max_8: Left_1[8] = MAX ( Left_0_aux_8 , 0 )
 Constr_Lost_1_max_8: Lost_1[8] = MAX ( Lost_0_aux_8 , 0 )
 logistic_constraint_9: Fraction_for_second_order_amount[9] =
   LOGISTIC ( f_var[9] )
 GC226: exp_tau_var_9_0 = EXP ( tau[9,0] )
 GC227: exp_tau_var_9_1 = EXP ( tau[9,1] )
 GC228: exp_tau_var_9_2 = EXP ( tau[9,2] )
 GC229: exp_tau_var_9_3 = EXP ( tau[9,3] )
 GC230: exp_tau_var_9_4 = EXP ( tau[9,4] )
 GC231: exp_tau_var_9_5 = EXP ( tau[9,5] )
 GC232: exp_tau_var_9_6 = EXP ( tau[9,6] )
 GC233: exp_tau_var_9_7 = EXP ( tau[9,7] )
 MaxRConstraint_9: max_r_helper = MAX ( r[9,0] , r[9,1] , r[9,2] , r[9,3] ,
   r[9,4] , r[9,5] , r[9,6] , r[9,7] )
 Max_Constraint_9: Q1_var[9] = MAX ( Q_hat_adjusted[9] , 0 )
 Constr_Sold_0_min_9: Sold_0[9] = MIN ( Total_Demand_Up_to_K_minus_1[9] ,
   Q0_var[9] )
 Constr_Left_0_max_9: Left_0[9] = MAX ( Left_0_aux_9 , 0 )
 Constr_Lost_0_max_9: Lost_0[9] = MAX ( Lost_0_aux_9 , 0 )
 Constr_Sold_1_min_9: Sold_1[9] = MIN ( Total_Demand_from_k_to_T[9] ,
   Q1_plus_left[9] )
 Constr_Left_1_max_9: Left_1[9] = MAX ( Left_0_aux_9 , 0 )
 Constr_Lost_1_max_9: Lost_1[9] = MAX ( Lost_0_aux_9 , 0 )
End
