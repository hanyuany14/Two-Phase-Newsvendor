\ Model profit_maximization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0 Holding_Cost_0[0] + 0 Holding_Cost_0[1] + 0 Holding_Cost_0[2]
   + 0 Holding_Cost_0[3] + 0 Holding_Cost_0[4] + 0 Holding_Cost_0[5]
   + 0 Holding_Cost_0[6] + 0 Holding_Cost_0[7] + 0 Holding_Cost_0[8]
   + 0 Holding_Cost_0[9] + 0 Holding_Cost_0[10] + 0 Holding_Cost_0[11]
   + 0 Holding_Cost_0[12] + 0 Holding_Cost_0[13] + 0 Holding_Cost_0[14]
   + 0 Holding_Cost_0[15] + 0 Holding_Cost_0[16] + 0 Holding_Cost_0[17]
   + 0 Holding_Cost_0[18] + 0 Holding_Cost_0[19] + 0 Holding_Cost_0[20]
   + 0 Holding_Cost_0[21] + 0 Holding_Cost_0[22] + 0 Holding_Cost_0[23]
   + 0 Holding_Cost_0[24] + 0 Holding_Cost_1[0] + 0 Holding_Cost_1[1]
   + 0 Holding_Cost_1[2] + 0 Holding_Cost_1[3] + 0 Holding_Cost_1[4]
   + 0 Holding_Cost_1[5] + 0 Holding_Cost_1[6] + 0 Holding_Cost_1[7]
   + 0 Holding_Cost_1[8] + 0 Holding_Cost_1[9] + 0 Holding_Cost_1[10]
   + 0 Holding_Cost_1[11] + 0 Holding_Cost_1[12] + 0 Holding_Cost_1[13]
   + 0 Holding_Cost_1[14] + 0 Holding_Cost_1[15] + 0 Holding_Cost_1[16]
   + 0 Holding_Cost_1[17] + 0 Holding_Cost_1[18] + 0 Holding_Cost_1[19]
   + 0 Holding_Cost_1[20] + 0 Holding_Cost_1[21] + 0 Holding_Cost_1[22]
   + 0 Holding_Cost_1[23] + 0 Holding_Cost_1[24] + profits_vars[0]
   + profits_vars[1] + profits_vars[2] + profits_vars[3] + profits_vars[4]
   + profits_vars[5] + profits_vars[6] + profits_vars[7] + profits_vars[8]
   + profits_vars[9] + profits_vars[10] + profits_vars[11]
   + profits_vars[12] + profits_vars[13] + profits_vars[14]
   + profits_vars[15] + profits_vars[16] + profits_vars[17]
   + profits_vars[18] + profits_vars[19] + profits_vars[20]
   + profits_vars[21] + profits_vars[22] + profits_vars[23]
   + profits_vars[24] + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper
Subject To
 R0: - 1.541253849845814 alphas[0] - 2.019990581899547 alphas[1]
   - 0.4082936743355209 alphas[2] - 0.1538371179486136 alphas[3]
   - alphas[4] + f_var[0] = 0
 Q0_upper_bound_0: - 1829.241320246946 Fraction_for_second_order_amount[0]
   + Q0_var[0] = 0
 tau_computation_0_0: - 1.541253849845814 betas[0,0]
   - 2.019990581899547 betas[0,1] - 0.4082936743355209 betas[0,2]
   - 0.1538371179486136 betas[0,3] - betas[0,4] + tau[0,0] = 0
 neg_tau_constr_0_0: tau[0,0] + neg_tau_var_0_0 = 0
 tau_computation_0_1: - 1.541253849845814 betas[1,0]
   - 2.019990581899547 betas[1,1] - 0.4082936743355209 betas[1,2]
   - 0.1538371179486136 betas[1,3] - betas[1,4] + tau[0,1] = 0
 neg_tau_constr_0_1: tau[0,1] + neg_tau_var_0_1 = 0
 tau_computation_0_2: - 1.541253849845814 betas[2,0]
   - 2.019990581899547 betas[2,1] - 0.4082936743355209 betas[2,2]
   - 0.1538371179486136 betas[2,3] - betas[2,4] + tau[0,2] = 0
 neg_tau_constr_0_2: tau[0,2] + neg_tau_var_0_2 = 0
 tau_computation_0_3: - 1.541253849845814 betas[3,0]
   - 2.019990581899547 betas[3,1] - 0.4082936743355209 betas[3,2]
   - 0.1538371179486136 betas[3,3] - betas[3,4] + tau[0,3] = 0
 neg_tau_constr_0_3: tau[0,3] + neg_tau_var_0_3 = 0
 tau_computation_0_4: - 1.541253849845814 betas[4,0]
   - 2.019990581899547 betas[4,1] - 0.4082936743355209 betas[4,2]
   - 0.1538371179486136 betas[4,3] - betas[4,4] + tau[0,4] = 0
 neg_tau_constr_0_4: tau[0,4] + neg_tau_var_0_4 = 0
 tau_computation_0_5: - 1.541253849845814 betas[5,0]
   - 2.019990581899547 betas[5,1] - 0.4082936743355209 betas[5,2]
   - 0.1538371179486136 betas[5,3] - betas[5,4] + tau[0,5] = 0
 neg_tau_constr_0_5: tau[0,5] + neg_tau_var_0_5 = 0
 tau_computation_0_6: - 1.541253849845814 betas[6,0]
   - 2.019990581899547 betas[6,1] - 0.4082936743355209 betas[6,2]
   - 0.1538371179486136 betas[6,3] - betas[6,4] + tau[0,6] = 0
 neg_tau_constr_0_6: tau[0,6] + neg_tau_var_0_6 = 0
 tau_computation_0_7: - 1.541253849845814 betas[7,0]
   - 2.019990581899547 betas[7,1] - 0.4082936743355209 betas[7,2]
   - 0.1538371179486136 betas[7,3] - betas[7,4] + tau[0,7] = 0
 neg_tau_constr_0_7: tau[0,7] + neg_tau_var_0_7 = 0
 sum_r_0: r[0,0] + r[0,1] + r[0,2] + r[0,3] + r[0,4] + r[0,5] + r[0,6]
   + r[0,7] = 1
 Ensure_only_one_R_true_0: R[0,0] + R[0,1] + R[0,2] + R[0,3] + R[0,4]
   + R[0,5] + R[0,6] + R[0,7] = 1
 Define_Q_hat_0: - 2344.147283187371 R[0,0] - 2296.481242496399 R[0,1]
   - 2299.940782629418 R[0,2] - 2300.364822516048 R[0,3]
   - 2294.15656264558 R[0,4] - 2295.001673364416 R[0,5]
   - 2293.068540901724 R[0,6] - 2288.185605884337 R[0,7] + Q_hat[0] = 0
 Adjust_Q_hat_0: Q0_var[0] - Q_hat[0] + Q_hat_adjusted[0] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_0: Total_Demand_Up_to_K_minus_1[0]
   - 234.6077081655532 R[0,0] - 463.6798686940772 R[0,1]
   - 693.1859094660481 R[0,2] - 923.5298743212443 R[0,3]
   - 1152.320891249775 R[0,4] - 1377.832875710965 R[0,5]
   - 1605.75649004091 R[0,6] - 1829.685420803608 R[0,7] = 0
 Constr_Total_Demand_from_K_to_T_0: Total_Demand_from_k_to_T[0]
   - 2060.660653604933 R[0,0] - 1831.58849307641 R[0,1]
   - 1602.082452304439 R[0,2] - 1371.738487449243 R[0,3]
   - 1142.947470520712 R[0,4] - 917.4354860595222 R[0,5]
   - 689.5118717295773 R[0,6] - 465.5829409668786 R[0,7] = 0
 Constr_Left_0_diff_aux_0: Sold_0[0] - Q0_var[0] + Left_0_aux_0 = 0
 Constr_Lost_0_diff_aux_0: - Total_Demand_Up_to_K_minus_1[0] + Q0_var[0]
   + Lost_0_aux_0 = 0
 Constr_Q1_plus_left_0: - Left_0[0] + Q1_plus_left[0] - Q1_var[0] = 0
 Constr_Left_1_diff_aux_0: Sold_1[0] - Q1_plus_left[0] + Left_0_aux_0 = 0
 Constr_Lost_1_diff_aux_0: - Total_Demand_from_k_to_T[0] + Q1_plus_left[0]
   + Lost_0_aux_0 = 0
 Profit_Constraint_0: - 500 Sold_0[0] + 500 Lost_0[0] - 500 Sold_1[0]
   + 500 Left_1[0] + 500 Lost_1[0] + profits_vars[0] = 0
 R30: 1.992898804628819 alphas[0] - 1.112594295343697 alphas[1]
   - 1.107356935299945 alphas[2] - 1.64244444220325 alphas[3] - alphas[4]
   + f_var[1] = 0
 Q0_upper_bound_1: - 1829.241320246946 Fraction_for_second_order_amount[1]
   + Q0_var[1] = 0
 tau_computation_1_0: 1.992898804628819 betas[0,0]
   - 1.112594295343697 betas[0,1] - 1.107356935299945 betas[0,2]
   - 1.64244444220325 betas[0,3] - betas[0,4] + tau[1,0] = 0
 neg_tau_constr_1_0: tau[1,0] + neg_tau_var_1_0 = 0
 tau_computation_1_1: 1.992898804628819 betas[1,0]
   - 1.112594295343697 betas[1,1] - 1.107356935299945 betas[1,2]
   - 1.64244444220325 betas[1,3] - betas[1,4] + tau[1,1] = 0
 neg_tau_constr_1_1: tau[1,1] + neg_tau_var_1_1 = 0
 tau_computation_1_2: 1.992898804628819 betas[2,0]
   - 1.112594295343697 betas[2,1] - 1.107356935299945 betas[2,2]
   - 1.64244444220325 betas[2,3] - betas[2,4] + tau[1,2] = 0
 neg_tau_constr_1_2: tau[1,2] + neg_tau_var_1_2 = 0
 tau_computation_1_3: 1.992898804628819 betas[3,0]
   - 1.112594295343697 betas[3,1] - 1.107356935299945 betas[3,2]
   - 1.64244444220325 betas[3,3] - betas[3,4] + tau[1,3] = 0
 neg_tau_constr_1_3: tau[1,3] + neg_tau_var_1_3 = 0
 tau_computation_1_4: 1.992898804628819 betas[4,0]
   - 1.112594295343697 betas[4,1] - 1.107356935299945 betas[4,2]
   - 1.64244444220325 betas[4,3] - betas[4,4] + tau[1,4] = 0
 neg_tau_constr_1_4: tau[1,4] + neg_tau_var_1_4 = 0
 tau_computation_1_5: 1.992898804628819 betas[5,0]
   - 1.112594295343697 betas[5,1] - 1.107356935299945 betas[5,2]
   - 1.64244444220325 betas[5,3] - betas[5,4] + tau[1,5] = 0
 neg_tau_constr_1_5: tau[1,5] + neg_tau_var_1_5 = 0
 tau_computation_1_6: 1.992898804628819 betas[6,0]
   - 1.112594295343697 betas[6,1] - 1.107356935299945 betas[6,2]
   - 1.64244444220325 betas[6,3] - betas[6,4] + tau[1,6] = 0
 neg_tau_constr_1_6: tau[1,6] + neg_tau_var_1_6 = 0
 tau_computation_1_7: 1.992898804628819 betas[7,0]
   - 1.112594295343697 betas[7,1] - 1.107356935299945 betas[7,2]
   - 1.64244444220325 betas[7,3] - betas[7,4] + tau[1,7] = 0
 neg_tau_constr_1_7: tau[1,7] + neg_tau_var_1_7 = 0
 sum_r_1: r[1,0] + r[1,1] + r[1,2] + r[1,3] + r[1,4] + r[1,5] + r[1,6]
   + r[1,7] = 1
 Ensure_only_one_R_true_1: R[1,0] + R[1,1] + R[1,2] + R[1,3] + R[1,4]
   + R[1,5] + R[1,6] + R[1,7] = 1
 Define_Q_hat_1: - 1883.837634987426 R[1,0] - 1901.684501409011 R[1,1]
   - 1890.795489315724 R[1,2] - 1910.127449983097 R[1,3]
   - 1895.608959503049 R[1,4] - 1901.929195905871 R[1,5]
   - 1908.884185010066 R[1,6] - 1912.187378370304 R[1,7] + Q_hat[1] = 0
 Adjust_Q_hat_1: Q0_var[1] - Q_hat[1] + Q_hat_adjusted[1] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_1: Total_Demand_Up_to_K_minus_1[1]
   - 189.5237454347192 R[1,0] - 382.1394150568279 R[1,1]
   - 567.8181246051131 R[1,2] - 760.7481360523475 R[1,3]
   - 942.1472487685405 R[1,4] - 1133.152251969025 R[1,5]
   - 1328.157446595987 R[1,6] - 1516.012594089963 R[1,7] = 0
 Constr_Total_Demand_from_K_to_T_1: Total_Demand_from_k_to_T[1]
   - 1725.739365157443 R[1,0] - 1533.123695535334 R[1,1]
   - 1347.444985987049 R[1,2] - 1154.514974539815 R[1,3]
   - 973.1158618236216 R[1,4] - 782.1108586231369 R[1,5]
   - 587.1056639961748 R[1,6] - 399.2505165021991 R[1,7] = 0
 Constr_Left_0_diff_aux_1: Sold_0[1] - Q0_var[1] + Left_0_aux_1 = 0
 Constr_Lost_0_diff_aux_1: - Total_Demand_Up_to_K_minus_1[1] + Q0_var[1]
   + Lost_0_aux_1 = 0
 Constr_Q1_plus_left_1: - Left_0[1] + Q1_plus_left[1] - Q1_var[1] = 0
 Constr_Left_1_diff_aux_1: Sold_1[1] - Q1_plus_left[1] + Left_0_aux_1 = 0
 Constr_Lost_1_diff_aux_1: - Total_Demand_from_k_to_T[1] + Q1_plus_left[1]
   + Lost_0_aux_1 = 0
 Profit_Constraint_1: - 500 Sold_0[1] + 500 Lost_0[1] - 500 Sold_1[1]
   + 500 Left_1[1] + 500 Lost_1[1] + profits_vars[1] = 0
 R60: - 1.483701890800565 alphas[0] - 0.2404841598113751 alphas[1]
   + 1.697978577377691 alphas[2] + 0.2465137999969989 alphas[3] - alphas[4]
   + f_var[2] = 0
 Q0_upper_bound_2: - 1829.241320246946 Fraction_for_second_order_amount[2]
   + Q0_var[2] = 0
 tau_computation_2_0: - 1.483701890800565 betas[0,0]
   - 0.2404841598113751 betas[0,1] + 1.697978577377691 betas[0,2]
   + 0.2465137999969989 betas[0,3] - betas[0,4] + tau[2,0] = 0
 neg_tau_constr_2_0: tau[2,0] + neg_tau_var_2_0 = 0
 tau_computation_2_1: - 1.483701890800565 betas[1,0]
   - 0.2404841598113751 betas[1,1] + 1.697978577377691 betas[1,2]
   + 0.2465137999969989 betas[1,3] - betas[1,4] + tau[2,1] = 0
 neg_tau_constr_2_1: tau[2,1] + neg_tau_var_2_1 = 0
 tau_computation_2_2: - 1.483701890800565 betas[2,0]
   - 0.2404841598113751 betas[2,1] + 1.697978577377691 betas[2,2]
   + 0.2465137999969989 betas[2,3] - betas[2,4] + tau[2,2] = 0
 neg_tau_constr_2_2: tau[2,2] + neg_tau_var_2_2 = 0
 tau_computation_2_3: - 1.483701890800565 betas[3,0]
   - 0.2404841598113751 betas[3,1] + 1.697978577377691 betas[3,2]
   + 0.2465137999969989 betas[3,3] - betas[3,4] + tau[2,3] = 0
 neg_tau_constr_2_3: tau[2,3] + neg_tau_var_2_3 = 0
 tau_computation_2_4: - 1.483701890800565 betas[4,0]
   - 0.2404841598113751 betas[4,1] + 1.697978577377691 betas[4,2]
   + 0.2465137999969989 betas[4,3] - betas[4,4] + tau[2,4] = 0
 neg_tau_constr_2_4: tau[2,4] + neg_tau_var_2_4 = 0
 tau_computation_2_5: - 1.483701890800565 betas[5,0]
   - 0.2404841598113751 betas[5,1] + 1.697978577377691 betas[5,2]
   + 0.2465137999969989 betas[5,3] - betas[5,4] + tau[2,5] = 0
 neg_tau_constr_2_5: tau[2,5] + neg_tau_var_2_5 = 0
 tau_computation_2_6: - 1.483701890800565 betas[6,0]
   - 0.2404841598113751 betas[6,1] + 1.697978577377691 betas[6,2]
   + 0.2465137999969989 betas[6,3] - betas[6,4] + tau[2,6] = 0
 neg_tau_constr_2_6: tau[2,6] + neg_tau_var_2_6 = 0
 tau_computation_2_7: - 1.483701890800565 betas[7,0]
   - 0.2404841598113751 betas[7,1] + 1.697978577377691 betas[7,2]
   + 0.2465137999969989 betas[7,3] - betas[7,4] + tau[2,7] = 0
 neg_tau_constr_2_7: tau[2,7] + neg_tau_var_2_7 = 0
 sum_r_2: r[2,0] + r[2,1] + r[2,2] + r[2,3] + r[2,4] + r[2,5] + r[2,6]
   + r[2,7] = 1
 Ensure_only_one_R_true_2: R[2,0] + R[2,1] + R[2,2] + R[2,3] + R[2,4]
   + R[2,5] + R[2,6] + R[2,7] = 1
 Define_Q_hat_2: - 2103.556102761394 R[2,0] - 2073.689510147432 R[2,1]
   - 2086.059088627125 R[2,2] - 2025.382040503008 R[2,3]
   - 2010.378343573173 R[2,4] - 2030.757427028251 R[2,5]
   - 2031.552215322269 R[2,6] - 2040.224290239275 R[2,7] + Q_hat[2] = 0
 Adjust_Q_hat_2: Q0_var[2] - Q_hat[2] + Q_hat_adjusted[2] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_2: Total_Demand_Up_to_K_minus_1[2]
   - 211.0435620432443 R[2,0] - 418.0836057943445 R[2,1]
   - 629.515295746112 R[2,2] - 829.3967362776755 R[2,3]
   - 1025.779054094571 R[2,4] - 1229.060853054206 R[2,5]
   - 1426.029505825622 R[2,6] - 1642.930953134821 R[2,7] = 0
 Constr_Total_Demand_from_K_to_T_2: Total_Demand_from_k_to_T[2]
   - 1826.572609273786 R[2,0] - 1619.532565522686 R[2,1]
   - 1408.100875570919 R[2,2] - 1208.219435039355 R[2,3]
   - 1011.83711722246 R[2,4] - 808.5553182628247 R[2,5]
   - 611.5866654914089 R[2,6] - 394.6852181822097 R[2,7] = 0
 Constr_Left_0_diff_aux_2: Sold_0[2] - Q0_var[2] + Left_0_aux_2 = 0
 Constr_Lost_0_diff_aux_2: - Total_Demand_Up_to_K_minus_1[2] + Q0_var[2]
   + Lost_0_aux_2 = 0
 Constr_Q1_plus_left_2: - Left_0[2] + Q1_plus_left[2] - Q1_var[2] = 0
 Constr_Left_1_diff_aux_2: Sold_1[2] - Q1_plus_left[2] + Left_0_aux_2 = 0
 Constr_Lost_1_diff_aux_2: - Total_Demand_from_k_to_T[2] + Q1_plus_left[2]
   + Lost_0_aux_2 = 0
 Profit_Constraint_2: - 500 Sold_0[2] + 500 Lost_0[2] - 500 Sold_1[2]
   + 500 Left_1[2] + 500 Lost_1[2] + profits_vars[2] = 0
 R90: - 1.337216120762022 alphas[0] + 0.9040950082146765 alphas[1]
   + 1.386108656938341 alphas[2] - 1.368202138528298 alphas[3] - alphas[4]
   + f_var[3] = 0
 Q0_upper_bound_3: - 1829.241320246946 Fraction_for_second_order_amount[3]
   + Q0_var[3] = 0
 tau_computation_3_0: - 1.337216120762022 betas[0,0]
   + 0.9040950082146765 betas[0,1] + 1.386108656938341 betas[0,2]
   - 1.368202138528298 betas[0,3] - betas[0,4] + tau[3,0] = 0
 neg_tau_constr_3_0: tau[3,0] + neg_tau_var_3_0 = 0
 tau_computation_3_1: - 1.337216120762022 betas[1,0]
   + 0.9040950082146765 betas[1,1] + 1.386108656938341 betas[1,2]
   - 1.368202138528298 betas[1,3] - betas[1,4] + tau[3,1] = 0
 neg_tau_constr_3_1: tau[3,1] + neg_tau_var_3_1 = 0
 tau_computation_3_2: - 1.337216120762022 betas[2,0]
   + 0.9040950082146765 betas[2,1] + 1.386108656938341 betas[2,2]
   - 1.368202138528298 betas[2,3] - betas[2,4] + tau[3,2] = 0
 neg_tau_constr_3_2: tau[3,2] + neg_tau_var_3_2 = 0
 tau_computation_3_3: - 1.337216120762022 betas[3,0]
   + 0.9040950082146765 betas[3,1] + 1.386108656938341 betas[3,2]
   - 1.368202138528298 betas[3,3] - betas[3,4] + tau[3,3] = 0
 neg_tau_constr_3_3: tau[3,3] + neg_tau_var_3_3 = 0
 tau_computation_3_4: - 1.337216120762022 betas[4,0]
   + 0.9040950082146765 betas[4,1] + 1.386108656938341 betas[4,2]
   - 1.368202138528298 betas[4,3] - betas[4,4] + tau[3,4] = 0
 neg_tau_constr_3_4: tau[3,4] + neg_tau_var_3_4 = 0
 tau_computation_3_5: - 1.337216120762022 betas[5,0]
   + 0.9040950082146765 betas[5,1] + 1.386108656938341 betas[5,2]
   - 1.368202138528298 betas[5,3] - betas[5,4] + tau[3,5] = 0
 neg_tau_constr_3_5: tau[3,5] + neg_tau_var_3_5 = 0
 tau_computation_3_6: - 1.337216120762022 betas[6,0]
   + 0.9040950082146765 betas[6,1] + 1.386108656938341 betas[6,2]
   - 1.368202138528298 betas[6,3] - betas[6,4] + tau[3,6] = 0
 neg_tau_constr_3_6: tau[3,6] + neg_tau_var_3_6 = 0
 tau_computation_3_7: - 1.337216120762022 betas[7,0]
   + 0.9040950082146765 betas[7,1] + 1.386108656938341 betas[7,2]
   - 1.368202138528298 betas[7,3] - betas[7,4] + tau[3,7] = 0
 neg_tau_constr_3_7: tau[3,7] + neg_tau_var_3_7 = 0
 sum_r_3: r[3,0] + r[3,1] + r[3,2] + r[3,3] + r[3,4] + r[3,5] + r[3,6]
   + r[3,7] = 1
 Ensure_only_one_R_true_3: R[3,0] + R[3,1] + R[3,2] + R[3,3] + R[3,4]
   + R[3,5] + R[3,6] + R[3,7] = 1
 Define_Q_hat_3: - 2125.03541304011 R[3,0] - 2130.649697876671 R[3,1]
   - 2129.117003277388 R[3,2] - 2133.710579718583 R[3,3]
   - 2136.167403810864 R[3,4] - 2125.593145879067 R[3,5]
   - 2120.452603686858 R[3,6] - 2118.780207666051 R[3,7] + Q_hat[3] = 0
 Adjust_Q_hat_3: Q0_var[3] - Q_hat[3] + Q_hat_adjusted[3] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_3: Total_Demand_Up_to_K_minus_1[3]
   - 213.147303357111 R[3,0] - 428.8667905459471 R[3,1]
   - 640.6981353300689 R[3,2] - 854.0792306780874 R[3,3]
   - 1066.101502879208 R[3,4] - 1272.788693010445 R[3,5]
   - 1481.329277770695 R[3,6] - 1690.211236960094 R[3,7] = 0
 Constr_Total_Demand_from_K_to_T_3: Total_Demand_from_k_to_T[3]
   - 1895.807085121464 R[3,0] - 1680.087597932628 R[3,1]
   - 1468.256253148506 R[3,2] - 1254.875157800487 R[3,3]
   - 1042.852885599367 R[3,4] - 836.1656954681299 R[3,5]
   - 627.6251107078799 R[3,6] - 418.7431515184811 R[3,7] = 0
 Constr_Left_0_diff_aux_3: Sold_0[3] - Q0_var[3] + Left_0_aux_3 = 0
 Constr_Lost_0_diff_aux_3: - Total_Demand_Up_to_K_minus_1[3] + Q0_var[3]
   + Lost_0_aux_3 = 0
 Constr_Q1_plus_left_3: - Left_0[3] + Q1_plus_left[3] - Q1_var[3] = 0
 Constr_Left_1_diff_aux_3: Sold_1[3] - Q1_plus_left[3] + Left_0_aux_3 = 0
 Constr_Lost_1_diff_aux_3: - Total_Demand_from_k_to_T[3] + Q1_plus_left[3]
   + Lost_0_aux_3 = 0
 Profit_Constraint_3: - 500 Sold_0[3] + 500 Lost_0[3] - 500 Sold_1[3]
   + 500 Left_1[3] + 500 Lost_1[3] + profits_vars[3] = 0
 R120: - 0.237881323713155 alphas[0] + 1.249411038760707 alphas[1]
   + 0.0902475462624092 alphas[2] + 0.8909197734323908 alphas[3]
   - alphas[4] + f_var[4] = 0
 Q0_upper_bound_4: - 1829.241320246946 Fraction_for_second_order_amount[4]
   + Q0_var[4] = 0
 tau_computation_4_0: - 0.237881323713155 betas[0,0]
   + 1.249411038760707 betas[0,1] + 0.0902475462624092 betas[0,2]
   + 0.8909197734323908 betas[0,3] - betas[0,4] + tau[4,0] = 0
 neg_tau_constr_4_0: tau[4,0] + neg_tau_var_4_0 = 0
 tau_computation_4_1: - 0.237881323713155 betas[1,0]
   + 1.249411038760707 betas[1,1] + 0.0902475462624092 betas[1,2]
   + 0.8909197734323908 betas[1,3] - betas[1,4] + tau[4,1] = 0
 neg_tau_constr_4_1: tau[4,1] + neg_tau_var_4_1 = 0
 tau_computation_4_2: - 0.237881323713155 betas[2,0]
   + 1.249411038760707 betas[2,1] + 0.0902475462624092 betas[2,2]
   + 0.8909197734323908 betas[2,3] - betas[2,4] + tau[4,2] = 0
 neg_tau_constr_4_2: tau[4,2] + neg_tau_var_4_2 = 0
 tau_computation_4_3: - 0.237881323713155 betas[3,0]
   + 1.249411038760707 betas[3,1] + 0.0902475462624092 betas[3,2]
   + 0.8909197734323908 betas[3,3] - betas[3,4] + tau[4,3] = 0
 neg_tau_constr_4_3: tau[4,3] + neg_tau_var_4_3 = 0
 tau_computation_4_4: - 0.237881323713155 betas[4,0]
   + 1.249411038760707 betas[4,1] + 0.0902475462624092 betas[4,2]
   + 0.8909197734323908 betas[4,3] - betas[4,4] + tau[4,4] = 0
 neg_tau_constr_4_4: tau[4,4] + neg_tau_var_4_4 = 0
 tau_computation_4_5: - 0.237881323713155 betas[5,0]
   + 1.249411038760707 betas[5,1] + 0.0902475462624092 betas[5,2]
   + 0.8909197734323908 betas[5,3] - betas[5,4] + tau[4,5] = 0
 neg_tau_constr_4_5: tau[4,5] + neg_tau_var_4_5 = 0
 tau_computation_4_6: - 0.237881323713155 betas[6,0]
   + 1.249411038760707 betas[6,1] + 0.0902475462624092 betas[6,2]
   + 0.8909197734323908 betas[6,3] - betas[6,4] + tau[4,6] = 0
 neg_tau_constr_4_6: tau[4,6] + neg_tau_var_4_6 = 0
 tau_computation_4_7: - 0.237881323713155 betas[7,0]
   + 1.249411038760707 betas[7,1] + 0.0902475462624092 betas[7,2]
   + 0.8909197734323908 betas[7,3] - betas[7,4] + tau[4,7] = 0
 neg_tau_constr_4_7: tau[4,7] + neg_tau_var_4_7 = 0
 sum_r_4: r[4,0] + r[4,1] + r[4,2] + r[4,3] + r[4,4] + r[4,5] + r[4,6]
   + r[4,7] = 1
 Ensure_only_one_R_true_4: R[4,0] + R[4,1] + R[4,2] + R[4,3] + R[4,4]
   + R[4,5] + R[4,6] + R[4,7] = 1
 Define_Q_hat_4: - 1512.160600993641 R[4,0] - 1527.693827910137 R[4,1]
   - 1539.118666478128 R[4,2] - 1574.960105184876 R[4,3]
   - 1579.584022342686 R[4,4] - 1576.433705823962 R[4,5]
   - 1576.287495857816 R[4,6] - 1581.615470746994 R[4,7] + Q_hat[4] = 0
 Adjust_Q_hat_4: Q0_var[4] - Q_hat[4] + Q_hat_adjusted[4] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_4: Total_Demand_Up_to_K_minus_1[4]
   - 153.1206988898331 R[4,0] - 306.3020039680786 R[4,1]
   - 464.2278475115409 R[4,2] - 622.4001019968608 R[4,3]
   - 780.0029340374053 R[4,4] - 938.7751729058556 R[4,5]
   - 1096.872045639398 R[4,6] - 1257.849020537764 R[4,7] = 0
 Constr_Total_Demand_from_K_to_T_4: Total_Demand_from_k_to_T[4]
   - 1432.588043089669 R[4,0] - 1279.406738011423 R[4,1]
   - 1121.480894467961 R[4,2] - 963.3086399826411 R[4,3]
   - 805.7058079420966 R[4,4] - 646.9335690736464 R[4,5]
   - 488.8366963401041 R[4,6] - 327.8597214417377 R[4,7] = 0
 Constr_Left_0_diff_aux_4: Sold_0[4] - Q0_var[4] + Left_0_aux_4 = 0
 Constr_Lost_0_diff_aux_4: - Total_Demand_Up_to_K_minus_1[4] + Q0_var[4]
   + Lost_0_aux_4 = 0
 Constr_Q1_plus_left_4: - Left_0[4] + Q1_plus_left[4] - Q1_var[4] = 0
 Constr_Left_1_diff_aux_4: Sold_1[4] - Q1_plus_left[4] + Left_0_aux_4 = 0
 Constr_Lost_1_diff_aux_4: - Total_Demand_from_k_to_T[4] + Q1_plus_left[4]
   + Lost_0_aux_4 = 0
 Profit_Constraint_4: - 500 Sold_0[4] + 500 Lost_0[4] - 500 Sold_1[4]
   + 500 Left_1[4] + 500 Lost_1[4] + profits_vars[4] = 0
 R150: - 0.2608669311813438 alphas[0] + 0.8234246579716354 alphas[1]
   + 0.2170664695893739 alphas[2] - 1.30359192641576 alphas[3] - alphas[4]
   + f_var[5] = 0
 Q0_upper_bound_5: - 1829.241320246946 Fraction_for_second_order_amount[5]
   + Q0_var[5] = 0
 tau_computation_5_0: - 0.2608669311813438 betas[0,0]
   + 0.8234246579716354 betas[0,1] + 0.2170664695893739 betas[0,2]
   - 1.30359192641576 betas[0,3] - betas[0,4] + tau[5,0] = 0
 neg_tau_constr_5_0: tau[5,0] + neg_tau_var_5_0 = 0
 tau_computation_5_1: - 0.2608669311813438 betas[1,0]
   + 0.8234246579716354 betas[1,1] + 0.2170664695893739 betas[1,2]
   - 1.30359192641576 betas[1,3] - betas[1,4] + tau[5,1] = 0
 neg_tau_constr_5_1: tau[5,1] + neg_tau_var_5_1 = 0
 tau_computation_5_2: - 0.2608669311813438 betas[2,0]
   + 0.8234246579716354 betas[2,1] + 0.2170664695893739 betas[2,2]
   - 1.30359192641576 betas[2,3] - betas[2,4] + tau[5,2] = 0
 neg_tau_constr_5_2: tau[5,2] + neg_tau_var_5_2 = 0
 tau_computation_5_3: - 0.2608669311813438 betas[3,0]
   + 0.8234246579716354 betas[3,1] + 0.2170664695893739 betas[3,2]
   - 1.30359192641576 betas[3,3] - betas[3,4] + tau[5,3] = 0
 neg_tau_constr_5_3: tau[5,3] + neg_tau_var_5_3 = 0
 tau_computation_5_4: - 0.2608669311813438 betas[4,0]
   + 0.8234246579716354 betas[4,1] + 0.2170664695893739 betas[4,2]
   - 1.30359192641576 betas[4,3] - betas[4,4] + tau[5,4] = 0
 neg_tau_constr_5_4: tau[5,4] + neg_tau_var_5_4 = 0
 tau_computation_5_5: - 0.2608669311813438 betas[5,0]
   + 0.8234246579716354 betas[5,1] + 0.2170664695893739 betas[5,2]
   - 1.30359192641576 betas[5,3] - betas[5,4] + tau[5,5] = 0
 neg_tau_constr_5_5: tau[5,5] + neg_tau_var_5_5 = 0
 tau_computation_5_6: - 0.2608669311813438 betas[6,0]
   + 0.8234246579716354 betas[6,1] + 0.2170664695893739 betas[6,2]
   - 1.30359192641576 betas[6,3] - betas[6,4] + tau[5,6] = 0
 neg_tau_constr_5_6: tau[5,6] + neg_tau_var_5_6 = 0
 tau_computation_5_7: - 0.2608669311813438 betas[7,0]
   + 0.8234246579716354 betas[7,1] + 0.2170664695893739 betas[7,2]
   - 1.30359192641576 betas[7,3] - betas[7,4] + tau[5,7] = 0
 neg_tau_constr_5_7: tau[5,7] + neg_tau_var_5_7 = 0
 sum_r_5: r[5,0] + r[5,1] + r[5,2] + r[5,3] + r[5,4] + r[5,5] + r[5,6]
   + r[5,7] = 1
 Ensure_only_one_R_true_5: R[5,0] + R[5,1] + R[5,2] + R[5,3] + R[5,4]
   + R[5,5] + R[5,6] + R[5,7] = 1
 Define_Q_hat_5: - 1946.463654505349 R[5,0] - 1972.95247569455 R[5,1]
   - 1979.990522964209 R[5,2] - 1955.903866504862 R[5,3]
   - 1949.666194708891 R[5,4] - 1949.02725202777 R[5,5]
   - 1947.987300566102 R[5,6] - 1946.604139485471 R[5,7] + Q_hat[5] = 0
 Adjust_Q_hat_5: Q0_var[5] - Q_hat[5] + Q_hat_adjusted[5] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_5: Total_Demand_Up_to_K_minus_1[5]
   - 195.6575059174021 R[5,0] - 396.4119598817947 R[5,1]
   - 596.4687178523036 R[5,2] - 789.9125765244125 R[5,3]
   - 980.5912640493669 R[5,4] - 1172.025384240986 R[5,5]
   - 1360.517533682113 R[5,6] - 1559.669583497634 R[5,7] = 0
 Constr_Total_Demand_from_K_to_T_5: Total_Demand_from_k_to_T[5]
   - 1745.038845353678 R[5,0] - 1544.284391389285 R[5,1]
   - 1344.227633418777 R[5,2] - 1150.783774746667 R[5,3]
   - 960.1050872217131 R[5,4] - 768.6709670300938 R[5,5]
   - 580.1788175889671 R[5,6] - 381.0267677734467 R[5,7] = 0
 Constr_Left_0_diff_aux_5: Sold_0[5] - Q0_var[5] + Left_0_aux_5 = 0
 Constr_Lost_0_diff_aux_5: - Total_Demand_Up_to_K_minus_1[5] + Q0_var[5]
   + Lost_0_aux_5 = 0
 Constr_Q1_plus_left_5: - Left_0[5] + Q1_plus_left[5] - Q1_var[5] = 0
 Constr_Left_1_diff_aux_5: Sold_1[5] - Q1_plus_left[5] + Left_0_aux_5 = 0
 Constr_Lost_1_diff_aux_5: - Total_Demand_from_k_to_T[5] + Q1_plus_left[5]
   + Lost_0_aux_5 = 0
 Profit_Constraint_5: - 500 Sold_0[5] + 500 Lost_0[5] - 500 Sold_1[5]
   + 500 Left_1[5] + 500 Lost_1[5] + profits_vars[5] = 0
 R180: - 0.9794235143501542 alphas[0] + 1.047396572258944 alphas[1]
   + 0.1956102706870147 alphas[2] + 0.4125050955682381 alphas[3]
   - alphas[4] + f_var[6] = 0
 Q0_upper_bound_6: - 1829.241320246946 Fraction_for_second_order_amount[6]
   + Q0_var[6] = 0
 tau_computation_6_0: - 0.9794235143501542 betas[0,0]
   + 1.047396572258944 betas[0,1] + 0.1956102706870147 betas[0,2]
   + 0.4125050955682381 betas[0,3] - betas[0,4] + tau[6,0] = 0
 neg_tau_constr_6_0: tau[6,0] + neg_tau_var_6_0 = 0
 tau_computation_6_1: - 0.9794235143501542 betas[1,0]
   + 1.047396572258944 betas[1,1] + 0.1956102706870147 betas[1,2]
   + 0.4125050955682381 betas[1,3] - betas[1,4] + tau[6,1] = 0
 neg_tau_constr_6_1: tau[6,1] + neg_tau_var_6_1 = 0
 tau_computation_6_2: - 0.9794235143501542 betas[2,0]
   + 1.047396572258944 betas[2,1] + 0.1956102706870147 betas[2,2]
   + 0.4125050955682381 betas[2,3] - betas[2,4] + tau[6,2] = 0
 neg_tau_constr_6_2: tau[6,2] + neg_tau_var_6_2 = 0
 tau_computation_6_3: - 0.9794235143501542 betas[3,0]
   + 1.047396572258944 betas[3,1] + 0.1956102706870147 betas[3,2]
   + 0.4125050955682381 betas[3,3] - betas[3,4] + tau[6,3] = 0
 neg_tau_constr_6_3: tau[6,3] + neg_tau_var_6_3 = 0
 tau_computation_6_4: - 0.9794235143501542 betas[4,0]
   + 1.047396572258944 betas[4,1] + 0.1956102706870147 betas[4,2]
   + 0.4125050955682381 betas[4,3] - betas[4,4] + tau[6,4] = 0
 neg_tau_constr_6_4: tau[6,4] + neg_tau_var_6_4 = 0
 tau_computation_6_5: - 0.9794235143501542 betas[5,0]
   + 1.047396572258944 betas[5,1] + 0.1956102706870147 betas[5,2]
   + 0.4125050955682381 betas[5,3] - betas[5,4] + tau[6,5] = 0
 neg_tau_constr_6_5: tau[6,5] + neg_tau_var_6_5 = 0
 tau_computation_6_6: - 0.9794235143501542 betas[6,0]
   + 1.047396572258944 betas[6,1] + 0.1956102706870147 betas[6,2]
   + 0.4125050955682381 betas[6,3] - betas[6,4] + tau[6,6] = 0
 neg_tau_constr_6_6: tau[6,6] + neg_tau_var_6_6 = 0
 tau_computation_6_7: - 0.9794235143501542 betas[7,0]
   + 1.047396572258944 betas[7,1] + 0.1956102706870147 betas[7,2]
   + 0.4125050955682381 betas[7,3] - betas[7,4] + tau[6,7] = 0
 neg_tau_constr_6_7: tau[6,7] + neg_tau_var_6_7 = 0
 sum_r_6: r[6,0] + r[6,1] + r[6,2] + r[6,3] + r[6,4] + r[6,5] + r[6,6]
   + r[6,7] = 1
 Ensure_only_one_R_true_6: R[6,0] + R[6,1] + R[6,2] + R[6,3] + R[6,4]
   + R[6,5] + R[6,6] + R[6,7] = 1
 Define_Q_hat_6: - 1713.611382841736 R[6,0] - 1756.127155760253 R[6,1]
   - 1760.456256858397 R[6,2] - 1787.81847262807 R[6,3]
   - 1787.917411190874 R[6,4] - 1805.302380032399 R[6,5]
   - 1801.396998771231 R[6,6] - 1800.006561218228 R[6,7] + Q_hat[6] = 0
 Adjust_Q_hat_6: Q0_var[6] - Q_hat[6] + Q_hat_adjusted[6] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_6: Total_Demand_Up_to_K_minus_1[6]
   - 172.8513292958069 R[6,0] - 352.0650733693554 R[6,1]
   - 529.9032733326933 R[6,2] - 708.6270492581801 R[6,3]
   - 884.5817003030572 R[6,4] - 1074.122011396899 R[6,5]
   - 1256.431082223862 R[6,6] - 1440.37041360398 R[6,7] = 0
 Constr_Total_Demand_from_K_to_T_6: Total_Demand_from_k_to_T[6]
   - 1628.44780803939 R[6,0] - 1449.234063965842 R[6,1]
   - 1271.395864002504 R[6,2] - 1092.672088077017 R[6,3]
   - 916.7174370321397 R[6,4] - 727.1771259382983 R[6,5]
   - 544.8680551113348 R[6,6] - 360.9287237312172 R[6,7] = 0
 Constr_Left_0_diff_aux_6: Sold_0[6] - Q0_var[6] + Left_0_aux_6 = 0
 Constr_Lost_0_diff_aux_6: - Total_Demand_Up_to_K_minus_1[6] + Q0_var[6]
   + Lost_0_aux_6 = 0
 Constr_Q1_plus_left_6: - Left_0[6] + Q1_plus_left[6] - Q1_var[6] = 0
 Constr_Left_1_diff_aux_6: Sold_1[6] - Q1_plus_left[6] + Left_0_aux_6 = 0
 Constr_Lost_1_diff_aux_6: - Total_Demand_from_k_to_T[6] + Q1_plus_left[6]
   + Lost_0_aux_6 = 0
 Profit_Constraint_6: - 500 Sold_0[6] + 500 Lost_0[6] - 500 Sold_1[6]
   + 500 Left_1[6] + 500 Lost_1[6] + profits_vars[6] = 0
 R210: 0.8412023743235226 alphas[0] + 0.2748705245454496 alphas[1]
   - 1.126748281959851 alphas[2] + 0.2780283782316132 alphas[3] - alphas[4]
   + f_var[7] = 0
 Q0_upper_bound_7: - 1829.241320246946 Fraction_for_second_order_amount[7]
   + Q0_var[7] = 0
 tau_computation_7_0: 0.8412023743235226 betas[0,0]
   + 0.2748705245454496 betas[0,1] - 1.126748281959851 betas[0,2]
   + 0.2780283782316132 betas[0,3] - betas[0,4] + tau[7,0] = 0
 neg_tau_constr_7_0: tau[7,0] + neg_tau_var_7_0 = 0
 tau_computation_7_1: 0.8412023743235226 betas[1,0]
   + 0.2748705245454496 betas[1,1] - 1.126748281959851 betas[1,2]
   + 0.2780283782316132 betas[1,3] - betas[1,4] + tau[7,1] = 0
 neg_tau_constr_7_1: tau[7,1] + neg_tau_var_7_1 = 0
 tau_computation_7_2: 0.8412023743235226 betas[2,0]
   + 0.2748705245454496 betas[2,1] - 1.126748281959851 betas[2,2]
   + 0.2780283782316132 betas[2,3] - betas[2,4] + tau[7,2] = 0
 neg_tau_constr_7_2: tau[7,2] + neg_tau_var_7_2 = 0
 tau_computation_7_3: 0.8412023743235226 betas[3,0]
   + 0.2748705245454496 betas[3,1] - 1.126748281959851 betas[3,2]
   + 0.2780283782316132 betas[3,3] - betas[3,4] + tau[7,3] = 0
 neg_tau_constr_7_3: tau[7,3] + neg_tau_var_7_3 = 0
 tau_computation_7_4: 0.8412023743235226 betas[4,0]
   + 0.2748705245454496 betas[4,1] - 1.126748281959851 betas[4,2]
   + 0.2780283782316132 betas[4,3] - betas[4,4] + tau[7,4] = 0
 neg_tau_constr_7_4: tau[7,4] + neg_tau_var_7_4 = 0
 tau_computation_7_5: 0.8412023743235226 betas[5,0]
   + 0.2748705245454496 betas[5,1] - 1.126748281959851 betas[5,2]
   + 0.2780283782316132 betas[5,3] - betas[5,4] + tau[7,5] = 0
 neg_tau_constr_7_5: tau[7,5] + neg_tau_var_7_5 = 0
 tau_computation_7_6: 0.8412023743235226 betas[6,0]
   + 0.2748705245454496 betas[6,1] - 1.126748281959851 betas[6,2]
   + 0.2780283782316132 betas[6,3] - betas[6,4] + tau[7,6] = 0
 neg_tau_constr_7_6: tau[7,6] + neg_tau_var_7_6 = 0
 tau_computation_7_7: 0.8412023743235226 betas[7,0]
   + 0.2748705245454496 betas[7,1] - 1.126748281959851 betas[7,2]
   + 0.2780283782316132 betas[7,3] - betas[7,4] + tau[7,7] = 0
 neg_tau_constr_7_7: tau[7,7] + neg_tau_var_7_7 = 0
 sum_r_7: r[7,0] + r[7,1] + r[7,2] + r[7,3] + r[7,4] + r[7,5] + r[7,6]
   + r[7,7] = 1
 Ensure_only_one_R_true_7: R[7,0] + R[7,1] + R[7,2] + R[7,3] + R[7,4]
   + R[7,5] + R[7,6] + R[7,7] = 1
 Define_Q_hat_7: - 1612.517456921152 R[7,0] - 1643.249328883402 R[7,1]
   - 1624.423245491607 R[7,2] - 1629.31614176033 R[7,3]
   - 1617.524887827012 R[7,4] - 1623.562340771564 R[7,5]
   - 1624.515385975614 R[7,6] - 1618.045786699398 R[7,7] + Q_hat[7] = 0
 Adjust_Q_hat_7: Q0_var[7] - Q_hat[7] + Q_hat_adjusted[7] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_7: Total_Demand_Up_to_K_minus_1[7]
   - 162.9499187858591 R[7,0] - 329.4179217225098 R[7,1]
   - 486.629904904724 R[7,2] - 652.3455267178776 R[7,3]
   - 803.0439490375285 R[7,4] - 965.8670471190485 R[7,5]
   - 1130.85189783895 R[7,6] - 1285.411795231588 R[7,7] = 0
 Constr_Total_Demand_from_K_to_T_7: Total_Demand_from_k_to_T[7]
   - 1456.850500960996 R[7,0] - 1290.382498024346 R[7,1]
   - 1133.170514842131 R[7,2] - 967.4548930289778 R[7,3]
   - 816.756470709327 R[7,4] - 653.933372627807 R[7,5]
   - 488.9485219079054 R[7,6] - 334.3886245152671 R[7,7] = 0
 Constr_Left_0_diff_aux_7: Sold_0[7] - Q0_var[7] + Left_0_aux_7 = 0
 Constr_Lost_0_diff_aux_7: - Total_Demand_Up_to_K_minus_1[7] + Q0_var[7]
   + Lost_0_aux_7 = 0
 Constr_Q1_plus_left_7: - Left_0[7] + Q1_plus_left[7] - Q1_var[7] = 0
 Constr_Left_1_diff_aux_7: Sold_1[7] - Q1_plus_left[7] + Left_0_aux_7 = 0
 Constr_Lost_1_diff_aux_7: - Total_Demand_from_k_to_T[7] + Q1_plus_left[7]
   + Lost_0_aux_7 = 0
 Profit_Constraint_7: - 500 Sold_0[7] + 500 Lost_0[7] - 500 Sold_1[7]
   + 500 Left_1[7] + 500 Lost_1[7] + profits_vars[7] = 0
 R240: 0.8250289196442896 alphas[0] + 1.369727524695598 alphas[1]
   - 1.555828833523332 alphas[2] + 0.1507512078129342 alphas[3] - alphas[4]
   + f_var[8] = 0
 Q0_upper_bound_8: - 1829.241320246946 Fraction_for_second_order_amount[8]
   + Q0_var[8] = 0
 tau_computation_8_0: 0.8250289196442896 betas[0,0]
   + 1.369727524695598 betas[0,1] - 1.555828833523332 betas[0,2]
   + 0.1507512078129342 betas[0,3] - betas[0,4] + tau[8,0] = 0
 neg_tau_constr_8_0: tau[8,0] + neg_tau_var_8_0 = 0
 tau_computation_8_1: 0.8250289196442896 betas[1,0]
   + 1.369727524695598 betas[1,1] - 1.555828833523332 betas[1,2]
   + 0.1507512078129342 betas[1,3] - betas[1,4] + tau[8,1] = 0
 neg_tau_constr_8_1: tau[8,1] + neg_tau_var_8_1 = 0
 tau_computation_8_2: 0.8250289196442896 betas[2,0]
   + 1.369727524695598 betas[2,1] - 1.555828833523332 betas[2,2]
   + 0.1507512078129342 betas[2,3] - betas[2,4] + tau[8,2] = 0
 neg_tau_constr_8_2: tau[8,2] + neg_tau_var_8_2 = 0
 tau_computation_8_3: 0.8250289196442896 betas[3,0]
   + 1.369727524695598 betas[3,1] - 1.555828833523332 betas[3,2]
   + 0.1507512078129342 betas[3,3] - betas[3,4] + tau[8,3] = 0
 neg_tau_constr_8_3: tau[8,3] + neg_tau_var_8_3 = 0
 tau_computation_8_4: 0.8250289196442896 betas[4,0]
   + 1.369727524695598 betas[4,1] - 1.555828833523332 betas[4,2]
   + 0.1507512078129342 betas[4,3] - betas[4,4] + tau[8,4] = 0
 neg_tau_constr_8_4: tau[8,4] + neg_tau_var_8_4 = 0
 tau_computation_8_5: 0.8250289196442896 betas[5,0]
   + 1.369727524695598 betas[5,1] - 1.555828833523332 betas[5,2]
   + 0.1507512078129342 betas[5,3] - betas[5,4] + tau[8,5] = 0
 neg_tau_constr_8_5: tau[8,5] + neg_tau_var_8_5 = 0
 tau_computation_8_6: 0.8250289196442896 betas[6,0]
   + 1.369727524695598 betas[6,1] - 1.555828833523332 betas[6,2]
   + 0.1507512078129342 betas[6,3] - betas[6,4] + tau[8,6] = 0
 neg_tau_constr_8_6: tau[8,6] + neg_tau_var_8_6 = 0
 tau_computation_8_7: 0.8250289196442896 betas[7,0]
   + 1.369727524695598 betas[7,1] - 1.555828833523332 betas[7,2]
   + 0.1507512078129342 betas[7,3] - betas[7,4] + tau[8,7] = 0
 neg_tau_constr_8_7: tau[8,7] + neg_tau_var_8_7 = 0
 sum_r_8: r[8,0] + r[8,1] + r[8,2] + r[8,3] + r[8,4] + r[8,5] + r[8,6]
   + r[8,7] = 1
 Ensure_only_one_R_true_8: R[8,0] + R[8,1] + R[8,2] + R[8,3] + R[8,4]
   + R[8,5] + R[8,6] + R[8,7] = 1
 Define_Q_hat_8: - 1602.915924325181 R[8,0] - 1518.299843449184 R[8,1]
   - 1527.785724545096 R[8,2] - 1534.502799504948 R[8,3]
   - 1521.850554273337 R[8,4] - 1508.860039421324 R[8,5]
   - 1518.364368275839 R[8,6] - 1512.298975027852 R[8,7] + Q_hat[8] = 0
 Adjust_Q_hat_8: Q0_var[8] - Q_hat[8] + Q_hat_adjusted[8] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_8: Total_Demand_Up_to_K_minus_1[8]
   - 162.0095189063249 R[8,0] - 306.5829002849281 R[8,1]
   - 461.5737248336549 R[8,2] - 617.8648816519404 R[8,3]
   - 766.9379880283843 R[8,4] - 907.8175798614313 R[8,5]
   - 1059.495672769851 R[8,6] - 1200.344547039831 R[8,7] = 0
 Constr_Total_Demand_from_K_to_T_8: Total_Demand_from_k_to_T[8]
   - 1350.544182242146 R[8,0] - 1205.970800863543 R[8,1]
   - 1050.979976314816 R[8,2] - 894.6888194965309 R[8,3]
   - 745.6157131200869 R[8,4] - 604.7361212870399 R[8,5]
   - 453.0580283786202 R[8,6] - 312.2091541086405 R[8,7] = 0
 Constr_Left_0_diff_aux_8: Sold_0[8] - Q0_var[8] + Left_0_aux_8 = 0
 Constr_Lost_0_diff_aux_8: - Total_Demand_Up_to_K_minus_1[8] + Q0_var[8]
   + Lost_0_aux_8 = 0
 Constr_Q1_plus_left_8: - Left_0[8] + Q1_plus_left[8] - Q1_var[8] = 0
 Constr_Left_1_diff_aux_8: Sold_1[8] - Q1_plus_left[8] + Left_0_aux_8 = 0
 Constr_Lost_1_diff_aux_8: - Total_Demand_from_k_to_T[8] + Q1_plus_left[8]
   + Lost_0_aux_8 = 0
 Profit_Constraint_8: - 500 Sold_0[8] + 500 Lost_0[8] - 500 Sold_1[8]
   + 500 Left_1[8] + 500 Lost_1[8] + profits_vars[8] = 0
 R270: - 0.5275284126148034 alphas[0] - 0.1438284086530026 alphas[1]
   + 1.501154537810019 alphas[2] - 1.374333751994383 alphas[3] - alphas[4]
   + f_var[9] = 0
 Q0_upper_bound_9: - 1829.241320246946 Fraction_for_second_order_amount[9]
   + Q0_var[9] = 0
 tau_computation_9_0: - 0.5275284126148034 betas[0,0]
   - 0.1438284086530026 betas[0,1] + 1.501154537810019 betas[0,2]
   - 1.374333751994383 betas[0,3] - betas[0,4] + tau[9,0] = 0
 neg_tau_constr_9_0: tau[9,0] + neg_tau_var_9_0 = 0
 tau_computation_9_1: - 0.5275284126148034 betas[1,0]
   - 0.1438284086530026 betas[1,1] + 1.501154537810019 betas[1,2]
   - 1.374333751994383 betas[1,3] - betas[1,4] + tau[9,1] = 0
 neg_tau_constr_9_1: tau[9,1] + neg_tau_var_9_1 = 0
 tau_computation_9_2: - 0.5275284126148034 betas[2,0]
   - 0.1438284086530026 betas[2,1] + 1.501154537810019 betas[2,2]
   - 1.374333751994383 betas[2,3] - betas[2,4] + tau[9,2] = 0
 neg_tau_constr_9_2: tau[9,2] + neg_tau_var_9_2 = 0
 tau_computation_9_3: - 0.5275284126148034 betas[3,0]
   - 0.1438284086530026 betas[3,1] + 1.501154537810019 betas[3,2]
   - 1.374333751994383 betas[3,3] - betas[3,4] + tau[9,3] = 0
 neg_tau_constr_9_3: tau[9,3] + neg_tau_var_9_3 = 0
 tau_computation_9_4: - 0.5275284126148034 betas[4,0]
   - 0.1438284086530026 betas[4,1] + 1.501154537810019 betas[4,2]
   - 1.374333751994383 betas[4,3] - betas[4,4] + tau[9,4] = 0
 neg_tau_constr_9_4: tau[9,4] + neg_tau_var_9_4 = 0
 tau_computation_9_5: - 0.5275284126148034 betas[5,0]
   - 0.1438284086530026 betas[5,1] + 1.501154537810019 betas[5,2]
   - 1.374333751994383 betas[5,3] - betas[5,4] + tau[9,5] = 0
 neg_tau_constr_9_5: tau[9,5] + neg_tau_var_9_5 = 0
 tau_computation_9_6: - 0.5275284126148034 betas[6,0]
   - 0.1438284086530026 betas[6,1] + 1.501154537810019 betas[6,2]
   - 1.374333751994383 betas[6,3] - betas[6,4] + tau[9,6] = 0
 neg_tau_constr_9_6: tau[9,6] + neg_tau_var_9_6 = 0
 tau_computation_9_7: - 0.5275284126148034 betas[7,0]
   - 0.1438284086530026 betas[7,1] + 1.501154537810019 betas[7,2]
   - 1.374333751994383 betas[7,3] - betas[7,4] + tau[9,7] = 0
 neg_tau_constr_9_7: tau[9,7] + neg_tau_var_9_7 = 0
 sum_r_9: r[9,0] + r[9,1] + r[9,2] + r[9,3] + r[9,4] + r[9,5] + r[9,6]
   + r[9,7] = 1
 Ensure_only_one_R_true_9: R[9,0] + R[9,1] + R[9,2] + R[9,3] + R[9,4]
   + R[9,5] + R[9,6] + R[9,7] = 1
 Define_Q_hat_9: - 2182.215054961522 R[9,0] - 2173.175513947964 R[9,1]
   - 2163.343489758653 R[9,2] - 2165.950061504669 R[9,3]
   - 2142.591002445306 R[9,4] - 2143.240275440236 R[9,5]
   - 2150.360036426043 R[9,6] - 2154.869305243548 R[9,7] + Q_hat[9] = 0
 Adjust_Q_hat_9: Q0_var[9] - Q_hat[9] + Q_hat_adjusted[9] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_9: Total_Demand_Up_to_K_minus_1[9]
   - 218.7476309961866 R[9,0] - 437.8158843924974 R[9,1]
   - 650.2850835098154 R[9,2] - 868.1740535605315 R[9,3]
   - 1073.718669467024 R[9,4] - 1283.230769174064 R[9,5]
   - 1499.37017989706 R[9,6] - 1710.162046923443 R[9,7] = 0
 Constr_Total_Demand_from_K_to_T_9: Total_Demand_from_k_to_T[9]
   - 1935.408020512661 R[9,0] - 1716.33976711635 R[9,1]
   - 1503.870567999032 R[9,2] - 1285.981597948316 R[9,3]
   - 1080.436982041823 R[9,4] - 870.9248823347832 R[9,5]
   - 654.7854716117866 R[9,6] - 443.9936045854041 R[9,7] = 0
 Constr_Left_0_diff_aux_9: Sold_0[9] - Q0_var[9] + Left_0_aux_9 = 0
 Constr_Lost_0_diff_aux_9: - Total_Demand_Up_to_K_minus_1[9] + Q0_var[9]
   + Lost_0_aux_9 = 0
 Constr_Q1_plus_left_9: - Left_0[9] + Q1_plus_left[9] - Q1_var[9] = 0
 Constr_Left_1_diff_aux_9: Sold_1[9] - Q1_plus_left[9] + Left_0_aux_9 = 0
 Constr_Lost_1_diff_aux_9: - Total_Demand_from_k_to_T[9] + Q1_plus_left[9]
   + Lost_0_aux_9 = 0
 Profit_Constraint_9: - 500 Sold_0[9] + 500 Lost_0[9] - 500 Sold_1[9]
   + 500 Left_1[9] + 500 Lost_1[9] + profits_vars[9] = 0
 R300: 0.038713139135696 alphas[0] - 0.9033099391806745 alphas[1]
   - 0.3001850844399854 alphas[2] - 0.8606952602454936 alphas[3]
   - alphas[4] + f_var[10] = 0
 Q0_upper_bound_10:
   - 1829.241320246946 Fraction_for_second_order_amount[10] + Q0_var[10]
   = 0
 tau_computation_10_0: 0.038713139135696 betas[0,0]
   - 0.9033099391806745 betas[0,1] - 0.3001850844399854 betas[0,2]
   - 0.8606952602454936 betas[0,3] - betas[0,4] + tau[10,0] = 0
 neg_tau_constr_10_0: tau[10,0] + neg_tau_var_10_0 = 0
 tau_computation_10_1: 0.038713139135696 betas[1,0]
   - 0.9033099391806745 betas[1,1] - 0.3001850844399854 betas[1,2]
   - 0.8606952602454936 betas[1,3] - betas[1,4] + tau[10,1] = 0
 neg_tau_constr_10_1: tau[10,1] + neg_tau_var_10_1 = 0
 tau_computation_10_2: 0.038713139135696 betas[2,0]
   - 0.9033099391806745 betas[2,1] - 0.3001850844399854 betas[2,2]
   - 0.8606952602454936 betas[2,3] - betas[2,4] + tau[10,2] = 0
 neg_tau_constr_10_2: tau[10,2] + neg_tau_var_10_2 = 0
 tau_computation_10_3: 0.038713139135696 betas[3,0]
   - 0.9033099391806745 betas[3,1] - 0.3001850844399854 betas[3,2]
   - 0.8606952602454936 betas[3,3] - betas[3,4] + tau[10,3] = 0
 neg_tau_constr_10_3: tau[10,3] + neg_tau_var_10_3 = 0
 tau_computation_10_4: 0.038713139135696 betas[4,0]
   - 0.9033099391806745 betas[4,1] - 0.3001850844399854 betas[4,2]
   - 0.8606952602454936 betas[4,3] - betas[4,4] + tau[10,4] = 0
 neg_tau_constr_10_4: tau[10,4] + neg_tau_var_10_4 = 0
 tau_computation_10_5: 0.038713139135696 betas[5,0]
   - 0.9033099391806745 betas[5,1] - 0.3001850844399854 betas[5,2]
   - 0.8606952602454936 betas[5,3] - betas[5,4] + tau[10,5] = 0
 neg_tau_constr_10_5: tau[10,5] + neg_tau_var_10_5 = 0
 tau_computation_10_6: 0.038713139135696 betas[6,0]
   - 0.9033099391806745 betas[6,1] - 0.3001850844399854 betas[6,2]
   - 0.8606952602454936 betas[6,3] - betas[6,4] + tau[10,6] = 0
 neg_tau_constr_10_6: tau[10,6] + neg_tau_var_10_6 = 0
 tau_computation_10_7: 0.038713139135696 betas[7,0]
   - 0.9033099391806745 betas[7,1] - 0.3001850844399854 betas[7,2]
   - 0.8606952602454936 betas[7,3] - betas[7,4] + tau[10,7] = 0
 neg_tau_constr_10_7: tau[10,7] + neg_tau_var_10_7 = 0
 sum_r_10: r[10,0] + r[10,1] + r[10,2] + r[10,3] + r[10,4] + r[10,5]
   + r[10,6] + r[10,7] = 1
 Ensure_only_one_R_true_10: R[10,0] + R[10,1] + R[10,2] + R[10,3] + R[10,4]
   + R[10,5] + R[10,6] + R[10,7] = 1
 Define_Q_hat_10: - 2001.451676210695 R[10,0] - 1997.947528777055 R[10,1]
   - 1998.448164185233 R[10,2] - 2043.345212564221 R[10,3]
   - 2046.894235590503 R[10,4] - 2044.585204081211 R[10,5]
   - 2062.370021378433 R[10,6] - 2052.62885978969 R[10,7] + Q_hat[10] = 0
 Adjust_Q_hat_10: Q0_var[10] - Q_hat[10] + Q_hat_adjusted[10] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_10: Total_Demand_Up_to_K_minus_1[10]
   - 201.0431803887078 R[10,0] - 402.1387940149189 R[10,1]
   - 601.6362752483046 R[10,2] - 807.6033964658985 R[10,3]
   - 1012.486472815474 R[10,4] - 1217.389448848966 R[10,5]
   - 1431.37719857693 R[10,6] - 1627.443863505028 R[10,7] = 0
 Constr_Total_Demand_from_K_to_T_10: Total_Demand_from_k_to_T[10]
   - 1847.418708682802 R[10,0] - 1646.323095056591 R[10,1]
   - 1446.825613823205 R[10,2] - 1240.858492605611 R[10,3]
   - 1035.975416256035 R[10,4] - 831.0724402225437 R[10,5]
   - 617.0846904945797 R[10,6] - 421.0180255664817 R[10,7] = 0
 Constr_Left_0_diff_aux_10: Sold_0[10] - Q0_var[10] + Left_0_aux_10 = 0
 Constr_Lost_0_diff_aux_10: - Total_Demand_Up_to_K_minus_1[10] + Q0_var[10]
   + Lost_0_aux_10 = 0
 Constr_Q1_plus_left_10: - Left_0[10] + Q1_plus_left[10] - Q1_var[10] = 0
 Constr_Left_1_diff_aux_10: Sold_1[10] - Q1_plus_left[10] + Left_0_aux_10
   = 0
 Constr_Lost_1_diff_aux_10: - Total_Demand_from_k_to_T[10]
   + Q1_plus_left[10] + Lost_0_aux_10 = 0
 Profit_Constraint_10: - 500 Sold_0[10] + 500 Lost_0[10] - 500 Sold_1[10]
   + 500 Left_1[10] + 500 Lost_1[10] + profits_vars[10] = 0
 R330: - 0.1494228261696841 alphas[0] + 1.058421988503832 alphas[1]
   - 0.1060784815895609 alphas[2] - 0.3960217080076868 alphas[3]
   - alphas[4] + f_var[11] = 0
 Q0_upper_bound_11:
   - 1829.241320246946 Fraction_for_second_order_amount[11] + Q0_var[11]
   = 0
 tau_computation_11_0: - 0.1494228261696841 betas[0,0]
   + 1.058421988503832 betas[0,1] - 0.1060784815895609 betas[0,2]
   - 0.3960217080076868 betas[0,3] - betas[0,4] + tau[11,0] = 0
 neg_tau_constr_11_0: tau[11,0] + neg_tau_var_11_0 = 0
 tau_computation_11_1: - 0.1494228261696841 betas[1,0]
   + 1.058421988503832 betas[1,1] - 0.1060784815895609 betas[1,2]
   - 0.3960217080076868 betas[1,3] - betas[1,4] + tau[11,1] = 0
 neg_tau_constr_11_1: tau[11,1] + neg_tau_var_11_1 = 0
 tau_computation_11_2: - 0.1494228261696841 betas[2,0]
   + 1.058421988503832 betas[2,1] - 0.1060784815895609 betas[2,2]
   - 0.3960217080076868 betas[2,3] - betas[2,4] + tau[11,2] = 0
 neg_tau_constr_11_2: tau[11,2] + neg_tau_var_11_2 = 0
 tau_computation_11_3: - 0.1494228261696841 betas[3,0]
   + 1.058421988503832 betas[3,1] - 0.1060784815895609 betas[3,2]
   - 0.3960217080076868 betas[3,3] - betas[3,4] + tau[11,3] = 0
 neg_tau_constr_11_3: tau[11,3] + neg_tau_var_11_3 = 0
 tau_computation_11_4: - 0.1494228261696841 betas[4,0]
   + 1.058421988503832 betas[4,1] - 0.1060784815895609 betas[4,2]
   - 0.3960217080076868 betas[4,3] - betas[4,4] + tau[11,4] = 0
 neg_tau_constr_11_4: tau[11,4] + neg_tau_var_11_4 = 0
 tau_computation_11_5: - 0.1494228261696841 betas[5,0]
   + 1.058421988503832 betas[5,1] - 0.1060784815895609 betas[5,2]
   - 0.3960217080076868 betas[5,3] - betas[5,4] + tau[11,5] = 0
 neg_tau_constr_11_5: tau[11,5] + neg_tau_var_11_5 = 0
 tau_computation_11_6: - 0.1494228261696841 betas[6,0]
   + 1.058421988503832 betas[6,1] - 0.1060784815895609 betas[6,2]
   - 0.3960217080076868 betas[6,3] - betas[6,4] + tau[11,6] = 0
 neg_tau_constr_11_6: tau[11,6] + neg_tau_var_11_6 = 0
 tau_computation_11_7: - 0.1494228261696841 betas[7,0]
   + 1.058421988503832 betas[7,1] - 0.1060784815895609 betas[7,2]
   - 0.3960217080076868 betas[7,3] - betas[7,4] + tau[11,7] = 0
 neg_tau_constr_11_7: tau[11,7] + neg_tau_var_11_7 = 0
 sum_r_11: r[11,0] + r[11,1] + r[11,2] + r[11,3] + r[11,4] + r[11,5]
   + r[11,6] + r[11,7] = 1
 Ensure_only_one_R_true_11: R[11,0] + R[11,1] + R[11,2] + R[11,3] + R[11,4]
   + R[11,5] + R[11,6] + R[11,7] = 1
 Define_Q_hat_11: - 1759.666764413576 R[11,0] - 1760.252249351859 R[11,1]
   - 1756.324688645341 R[11,2] - 1775.181620514023 R[11,3]
   - 1772.554693669662 R[11,4] - 1777.218534097491 R[11,5]
   - 1772.798405891717 R[11,6] - 1780.298433265899 R[11,7] + Q_hat[11] = 0
 Adjust_Q_hat_11: Q0_var[11] - Q_hat[11] + Q_hat_adjusted[11] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_11: Total_Demand_Up_to_K_minus_1[11]
   - 177.3621170111173 R[11,0] - 353.8177888915291 R[11,1]
   - 528.2020328305802 R[11,2] - 707.6029498178425 R[11,3]
   - 880.6607488212539 R[11,4] - 1059.599308747974 R[11,5]
   - 1238.17276556269 R[11,6] - 1416.814567649922 R[11,7] = 0
 Constr_Total_Demand_from_K_to_T_11: Total_Demand_from_k_to_T[11]
   - 1612.418962385028 R[11,0] - 1435.963290504616 R[11,1]
   - 1261.579046565565 R[11,2] - 1082.178129578303 R[11,3]
   - 909.1203305748913 R[11,4] - 730.1817706481713 R[11,5]
   - 551.6083138334554 R[11,6] - 372.9665117462234 R[11,7] = 0
 Constr_Left_0_diff_aux_11: Sold_0[11] - Q0_var[11] + Left_0_aux_11 = 0
 Constr_Lost_0_diff_aux_11: - Total_Demand_Up_to_K_minus_1[11] + Q0_var[11]
   + Lost_0_aux_11 = 0
 Constr_Q1_plus_left_11: - Left_0[11] + Q1_plus_left[11] - Q1_var[11] = 0
 Constr_Left_1_diff_aux_11: Sold_1[11] - Q1_plus_left[11] + Left_0_aux_11
   = 0
 Constr_Lost_1_diff_aux_11: - Total_Demand_from_k_to_T[11]
   + Q1_plus_left[11] + Lost_0_aux_11 = 0
 Profit_Constraint_11: - 500 Sold_0[11] + 500 Lost_0[11] - 500 Sold_1[11]
   + 500 Left_1[11] + 500 Lost_1[11] + profits_vars[11] = 0
 R360: 1.769515248111474 alphas[0] - 1.192601218300086 alphas[1]
   + 0.8558708337472838 alphas[2] - 1.112615756120183 alphas[3] - alphas[4]
   + f_var[12] = 0
 Q0_upper_bound_12:
   - 1829.241320246946 Fraction_for_second_order_amount[12] + Q0_var[12]
   = 0
 tau_computation_12_0: 1.769515248111474 betas[0,0]
   - 1.192601218300086 betas[0,1] + 0.8558708337472838 betas[0,2]
   - 1.112615756120183 betas[0,3] - betas[0,4] + tau[12,0] = 0
 neg_tau_constr_12_0: tau[12,0] + neg_tau_var_12_0 = 0
 tau_computation_12_1: 1.769515248111474 betas[1,0]
   - 1.192601218300086 betas[1,1] + 0.8558708337472838 betas[1,2]
   - 1.112615756120183 betas[1,3] - betas[1,4] + tau[12,1] = 0
 neg_tau_constr_12_1: tau[12,1] + neg_tau_var_12_1 = 0
 tau_computation_12_2: 1.769515248111474 betas[2,0]
   - 1.192601218300086 betas[2,1] + 0.8558708337472838 betas[2,2]
   - 1.112615756120183 betas[2,3] - betas[2,4] + tau[12,2] = 0
 neg_tau_constr_12_2: tau[12,2] + neg_tau_var_12_2 = 0
 tau_computation_12_3: 1.769515248111474 betas[3,0]
   - 1.192601218300086 betas[3,1] + 0.8558708337472838 betas[3,2]
   - 1.112615756120183 betas[3,3] - betas[3,4] + tau[12,3] = 0
 neg_tau_constr_12_3: tau[12,3] + neg_tau_var_12_3 = 0
 tau_computation_12_4: 1.769515248111474 betas[4,0]
   - 1.192601218300086 betas[4,1] + 0.8558708337472838 betas[4,2]
   - 1.112615756120183 betas[4,3] - betas[4,4] + tau[12,4] = 0
 neg_tau_constr_12_4: tau[12,4] + neg_tau_var_12_4 = 0
 tau_computation_12_5: 1.769515248111474 betas[5,0]
   - 1.192601218300086 betas[5,1] + 0.8558708337472838 betas[5,2]
   - 1.112615756120183 betas[5,3] - betas[5,4] + tau[12,5] = 0
 neg_tau_constr_12_5: tau[12,5] + neg_tau_var_12_5 = 0
 tau_computation_12_6: 1.769515248111474 betas[6,0]
   - 1.192601218300086 betas[6,1] + 0.8558708337472838 betas[6,2]
   - 1.112615756120183 betas[6,3] - betas[6,4] + tau[12,6] = 0
 neg_tau_constr_12_6: tau[12,6] + neg_tau_var_12_6 = 0
 tau_computation_12_7: 1.769515248111474 betas[7,0]
   - 1.192601218300086 betas[7,1] + 0.8558708337472838 betas[7,2]
   - 1.112615756120183 betas[7,3] - betas[7,4] + tau[12,7] = 0
 neg_tau_constr_12_7: tau[12,7] + neg_tau_var_12_7 = 0
 sum_r_12: r[12,0] + r[12,1] + r[12,2] + r[12,3] + r[12,4] + r[12,5]
   + r[12,6] + r[12,7] = 1
 Ensure_only_one_R_true_12: R[12,0] + R[12,1] + R[12,2] + R[12,3] + R[12,4]
   + R[12,5] + R[12,6] + R[12,7] = 1
 Define_Q_hat_12: - 1942.652023405692 R[12,0] - 1943.874241607492 R[12,1]
   - 1939.612861286936 R[12,2] - 1939.669972922548 R[12,3]
   - 1946.237167815532 R[12,4] - 1930.446759163163 R[12,5]
   - 1919.543371427922 R[12,6] - 1923.454291159974 R[12,7] + Q_hat[12] = 0
 Adjust_Q_hat_12: Q0_var[12] - Q_hat[12] + Q_hat_adjusted[12] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_12: Total_Demand_Up_to_K_minus_1[12]
   - 195.2841845344752 R[12,0] - 391.0633533572441 R[12,1]
   - 583.3725495994961 R[12,2] - 778.1678016559587 R[12,3]
   - 970.7779977258596 R[12,4] - 1155.899387651031 R[12,5]
   - 1342.625964133336 R[12,6] - 1533.239342198989 R[12,7] = 0
 Constr_Total_Demand_from_K_to_T_12: Total_Demand_from_k_to_T[12]
   - 1727.36749187385 R[12,0] - 1531.588323051081 R[12,1]
   - 1339.279126808829 R[12,2] - 1144.483874752367 R[12,3]
   - 951.8736786824658 R[12,4] - 766.7522887572948 R[12,5]
   - 580.0257122749896 R[12,6] - 389.4123342093365 R[12,7] = 0
 Constr_Left_0_diff_aux_12: Sold_0[12] - Q0_var[12] + Left_0_aux_12 = 0
 Constr_Lost_0_diff_aux_12: - Total_Demand_Up_to_K_minus_1[12] + Q0_var[12]
   + Lost_0_aux_12 = 0
 Constr_Q1_plus_left_12: - Left_0[12] + Q1_plus_left[12] - Q1_var[12] = 0
 Constr_Left_1_diff_aux_12: Sold_1[12] - Q1_plus_left[12] + Left_0_aux_12
   = 0
 Constr_Lost_1_diff_aux_12: - Total_Demand_from_k_to_T[12]
   + Q1_plus_left[12] + Lost_0_aux_12 = 0
 Profit_Constraint_12: - 500 Sold_0[12] + 500 Lost_0[12] - 500 Sold_1[12]
   + 500 Left_1[12] + 500 Lost_1[12] + profits_vars[12] = 0
 R390: 0.5407381482136011 alphas[0] - 0.8920734897627978 alphas[1]
   + 0.1504414039447525 alphas[2] - 1.160513196598033 alphas[3] - alphas[4]
   + f_var[13] = 0
 Q0_upper_bound_13:
   - 1829.241320246946 Fraction_for_second_order_amount[13] + Q0_var[13]
   = 0
 tau_computation_13_0: 0.5407381482136011 betas[0,0]
   - 0.8920734897627978 betas[0,1] + 0.1504414039447525 betas[0,2]
   - 1.160513196598033 betas[0,3] - betas[0,4] + tau[13,0] = 0
 neg_tau_constr_13_0: tau[13,0] + neg_tau_var_13_0 = 0
 tau_computation_13_1: 0.5407381482136011 betas[1,0]
   - 0.8920734897627978 betas[1,1] + 0.1504414039447525 betas[1,2]
   - 1.160513196598033 betas[1,3] - betas[1,4] + tau[13,1] = 0
 neg_tau_constr_13_1: tau[13,1] + neg_tau_var_13_1 = 0
 tau_computation_13_2: 0.5407381482136011 betas[2,0]
   - 0.8920734897627978 betas[2,1] + 0.1504414039447525 betas[2,2]
   - 1.160513196598033 betas[2,3] - betas[2,4] + tau[13,2] = 0
 neg_tau_constr_13_2: tau[13,2] + neg_tau_var_13_2 = 0
 tau_computation_13_3: 0.5407381482136011 betas[3,0]
   - 0.8920734897627978 betas[3,1] + 0.1504414039447525 betas[3,2]
   - 1.160513196598033 betas[3,3] - betas[3,4] + tau[13,3] = 0
 neg_tau_constr_13_3: tau[13,3] + neg_tau_var_13_3 = 0
 tau_computation_13_4: 0.5407381482136011 betas[4,0]
   - 0.8920734897627978 betas[4,1] + 0.1504414039447525 betas[4,2]
   - 1.160513196598033 betas[4,3] - betas[4,4] + tau[13,4] = 0
 neg_tau_constr_13_4: tau[13,4] + neg_tau_var_13_4 = 0
 tau_computation_13_5: 0.5407381482136011 betas[5,0]
   - 0.8920734897627978 betas[5,1] + 0.1504414039447525 betas[5,2]
   - 1.160513196598033 betas[5,3] - betas[5,4] + tau[13,5] = 0
 neg_tau_constr_13_5: tau[13,5] + neg_tau_var_13_5 = 0
 tau_computation_13_6: 0.5407381482136011 betas[6,0]
   - 0.8920734897627978 betas[6,1] + 0.1504414039447525 betas[6,2]
   - 1.160513196598033 betas[6,3] - betas[6,4] + tau[13,6] = 0
 neg_tau_constr_13_6: tau[13,6] + neg_tau_var_13_6 = 0
 tau_computation_13_7: 0.5407381482136011 betas[7,0]
   - 0.8920734897627978 betas[7,1] + 0.1504414039447525 betas[7,2]
   - 1.160513196598033 betas[7,3] - betas[7,4] + tau[13,7] = 0
 neg_tau_constr_13_7: tau[13,7] + neg_tau_var_13_7 = 0
 sum_r_13: r[13,0] + r[13,1] + r[13,2] + r[13,3] + r[13,4] + r[13,5]
   + r[13,6] + r[13,7] = 1
 Ensure_only_one_R_true_13: R[13,0] + R[13,1] + R[13,2] + R[13,3] + R[13,4]
   + R[13,5] + R[13,6] + R[13,7] = 1
 Define_Q_hat_13: - 1938.124898706827 R[13,0] - 2020.732690902529 R[13,1]
   - 2030.869917908554 R[13,2] - 2041.735999188605 R[13,3]
   - 2037.732302458207 R[13,4] - 2025.317456130789 R[13,5]
   - 2015.630800943809 R[13,6] - 2010.66149003091 R[13,7] + Q_hat[13] = 0
 Adjust_Q_hat_13: Q0_var[13] - Q_hat[13] + Q_hat_adjusted[13] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_13: Total_Demand_Up_to_K_minus_1[13]
   - 194.8407857875673 R[13,0] - 404.8817038309907 R[13,1]
   - 611.5787409875725 R[13,2] - 812.6146408232993 R[13,3]
   - 1013.151980808254 R[13,4] - 1211.689882307673 R[13,5]
   - 1404.58856294832 R[13,6] - 1608.32400716075 R[13,7] = 0
 Constr_Total_Demand_from_K_to_T_13: Total_Demand_from_k_to_T[13]
   - 1821.279705329967 R[13,0] - 1611.238787286544 R[13,1]
   - 1404.541750129962 R[13,2] - 1203.505850294236 R[13,3]
   - 1002.968510309281 R[13,4] - 804.4306088098621 R[13,5]
   - 611.5319281692153 R[13,6] - 407.7964839567852 R[13,7] = 0
 Constr_Left_0_diff_aux_13: Sold_0[13] - Q0_var[13] + Left_0_aux_13 = 0
 Constr_Lost_0_diff_aux_13: - Total_Demand_Up_to_K_minus_1[13] + Q0_var[13]
   + Lost_0_aux_13 = 0
 Constr_Q1_plus_left_13: - Left_0[13] + Q1_plus_left[13] - Q1_var[13] = 0
 Constr_Left_1_diff_aux_13: Sold_1[13] - Q1_plus_left[13] + Left_0_aux_13
   = 0
 Constr_Lost_1_diff_aux_13: - Total_Demand_from_k_to_T[13]
   + Q1_plus_left[13] + Lost_0_aux_13 = 0
 Profit_Constraint_13: - 500 Sold_0[13] + 500 Lost_0[13] - 500 Sold_1[13]
   + 500 Left_1[13] + 500 Lost_1[13] + profits_vars[13] = 0
 R420: - 0.7789685860329789 alphas[0] - 0.7237226264797559 alphas[1]
   - 0.0674462009232731 alphas[2] + 1.613193352825219 alphas[3] - alphas[4]
   + f_var[14] = 0
 Q0_upper_bound_14:
   - 1829.241320246946 Fraction_for_second_order_amount[14] + Q0_var[14]
   = 0
 tau_computation_14_0: - 0.7789685860329789 betas[0,0]
   - 0.7237226264797559 betas[0,1] - 0.0674462009232731 betas[0,2]
   + 1.613193352825219 betas[0,3] - betas[0,4] + tau[14,0] = 0
 neg_tau_constr_14_0: tau[14,0] + neg_tau_var_14_0 = 0
 tau_computation_14_1: - 0.7789685860329789 betas[1,0]
   - 0.7237226264797559 betas[1,1] - 0.0674462009232731 betas[1,2]
   + 1.613193352825219 betas[1,3] - betas[1,4] + tau[14,1] = 0
 neg_tau_constr_14_1: tau[14,1] + neg_tau_var_14_1 = 0
 tau_computation_14_2: - 0.7789685860329789 betas[2,0]
   - 0.7237226264797559 betas[2,1] - 0.0674462009232731 betas[2,2]
   + 1.613193352825219 betas[2,3] - betas[2,4] + tau[14,2] = 0
 neg_tau_constr_14_2: tau[14,2] + neg_tau_var_14_2 = 0
 tau_computation_14_3: - 0.7789685860329789 betas[3,0]
   - 0.7237226264797559 betas[3,1] - 0.0674462009232731 betas[3,2]
   + 1.613193352825219 betas[3,3] - betas[3,4] + tau[14,3] = 0
 neg_tau_constr_14_3: tau[14,3] + neg_tau_var_14_3 = 0
 tau_computation_14_4: - 0.7789685860329789 betas[4,0]
   - 0.7237226264797559 betas[4,1] - 0.0674462009232731 betas[4,2]
   + 1.613193352825219 betas[4,3] - betas[4,4] + tau[14,4] = 0
 neg_tau_constr_14_4: tau[14,4] + neg_tau_var_14_4 = 0
 tau_computation_14_5: - 0.7789685860329789 betas[5,0]
   - 0.7237226264797559 betas[5,1] - 0.0674462009232731 betas[5,2]
   + 1.613193352825219 betas[5,3] - betas[5,4] + tau[14,5] = 0
 neg_tau_constr_14_5: tau[14,5] + neg_tau_var_14_5 = 0
 tau_computation_14_6: - 0.7789685860329789 betas[6,0]
   - 0.7237226264797559 betas[6,1] - 0.0674462009232731 betas[6,2]
   + 1.613193352825219 betas[6,3] - betas[6,4] + tau[14,6] = 0
 neg_tau_constr_14_6: tau[14,6] + neg_tau_var_14_6 = 0
 tau_computation_14_7: - 0.7789685860329789 betas[7,0]
   - 0.7237226264797559 betas[7,1] - 0.0674462009232731 betas[7,2]
   + 1.613193352825219 betas[7,3] - betas[7,4] + tau[14,7] = 0
 neg_tau_constr_14_7: tau[14,7] + neg_tau_var_14_7 = 0
 sum_r_14: r[14,0] + r[14,1] + r[14,2] + r[14,3] + r[14,4] + r[14,5]
   + r[14,6] + r[14,7] = 1
 Ensure_only_one_R_true_14: R[14,0] + R[14,1] + R[14,2] + R[14,3] + R[14,4]
   + R[14,5] + R[14,6] + R[14,7] = 1
 Define_Q_hat_14: - 1839.473898584894 R[14,0] - 1810.463579375596 R[14,1]
   - 1799.896062734699 R[14,2] - 1800.60567188725 R[14,3]
   - 1775.121288299514 R[14,4] - 1776.24954441845 R[14,5]
   - 1760.538945748325 R[14,6] - 1759.845560689608 R[14,7] + Q_hat[14] = 0
 Adjust_Q_hat_14: Q0_var[14] - Q_hat[14] + Q_hat_adjusted[14] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_14: Total_Demand_Up_to_K_minus_1[14]
   - 185.1786419888242 R[14,0] - 364.6526848666834 R[14,1]
   - 540.9200435344501 R[14,2] - 723.8274109303486 R[14,3]
   - 890.88996256169 R[14,4] - 1063.370957621275 R[14,5]
   - 1235.748844859031 R[14,6] - 1403.901481367409 R[14,7] = 0
 Constr_Total_Demand_from_K_to_T_14: Total_Demand_from_k_to_T[14]
   - 1569.278078255323 R[14,0] - 1389.804035377464 R[14,1]
   - 1213.536676709698 R[14,2] - 1030.629309313799 R[14,3]
   - 863.5667576824577 R[14,4] - 691.0857626228732 R[14,5]
   - 518.7078753851169 R[14,6] - 350.5552388767384 R[14,7] = 0
 Constr_Left_0_diff_aux_14: Sold_0[14] - Q0_var[14] + Left_0_aux_14 = 0
 Constr_Lost_0_diff_aux_14: - Total_Demand_Up_to_K_minus_1[14] + Q0_var[14]
   + Lost_0_aux_14 = 0
 Constr_Q1_plus_left_14: - Left_0[14] + Q1_plus_left[14] - Q1_var[14] = 0
 Constr_Left_1_diff_aux_14: Sold_1[14] - Q1_plus_left[14] + Left_0_aux_14
   = 0
 Constr_Lost_1_diff_aux_14: - Total_Demand_from_k_to_T[14]
   + Q1_plus_left[14] + Lost_0_aux_14 = 0
 Profit_Constraint_14: - 500 Sold_0[14] + 500 Lost_0[14] - 500 Sold_1[14]
   + 500 Left_1[14] + 500 Lost_1[14] + profits_vars[14] = 0
 R450: - 0.1308042153486798 alphas[0] - 0.4390934242476194 alphas[1]
   + 0.224750820811735 alphas[2] - 0.2051628180813711 alphas[3] - alphas[4]
   + f_var[15] = 0
 Q0_upper_bound_15:
   - 1829.241320246946 Fraction_for_second_order_amount[15] + Q0_var[15]
   = 0
 tau_computation_15_0: - 0.1308042153486798 betas[0,0]
   - 0.4390934242476194 betas[0,1] + 0.224750820811735 betas[0,2]
   - 0.2051628180813711 betas[0,3] - betas[0,4] + tau[15,0] = 0
 neg_tau_constr_15_0: tau[15,0] + neg_tau_var_15_0 = 0
 tau_computation_15_1: - 0.1308042153486798 betas[1,0]
   - 0.4390934242476194 betas[1,1] + 0.224750820811735 betas[1,2]
   - 0.2051628180813711 betas[1,3] - betas[1,4] + tau[15,1] = 0
 neg_tau_constr_15_1: tau[15,1] + neg_tau_var_15_1 = 0
 tau_computation_15_2: - 0.1308042153486798 betas[2,0]
   - 0.4390934242476194 betas[2,1] + 0.224750820811735 betas[2,2]
   - 0.2051628180813711 betas[2,3] - betas[2,4] + tau[15,2] = 0
 neg_tau_constr_15_2: tau[15,2] + neg_tau_var_15_2 = 0
 tau_computation_15_3: - 0.1308042153486798 betas[3,0]
   - 0.4390934242476194 betas[3,1] + 0.224750820811735 betas[3,2]
   - 0.2051628180813711 betas[3,3] - betas[3,4] + tau[15,3] = 0
 neg_tau_constr_15_3: tau[15,3] + neg_tau_var_15_3 = 0
 tau_computation_15_4: - 0.1308042153486798 betas[4,0]
   - 0.4390934242476194 betas[4,1] + 0.224750820811735 betas[4,2]
   - 0.2051628180813711 betas[4,3] - betas[4,4] + tau[15,4] = 0
 neg_tau_constr_15_4: tau[15,4] + neg_tau_var_15_4 = 0
 tau_computation_15_5: - 0.1308042153486798 betas[5,0]
   - 0.4390934242476194 betas[5,1] + 0.224750820811735 betas[5,2]
   - 0.2051628180813711 betas[5,3] - betas[5,4] + tau[15,5] = 0
 neg_tau_constr_15_5: tau[15,5] + neg_tau_var_15_5 = 0
 tau_computation_15_6: - 0.1308042153486798 betas[6,0]
   - 0.4390934242476194 betas[6,1] + 0.224750820811735 betas[6,2]
   - 0.2051628180813711 betas[6,3] - betas[6,4] + tau[15,6] = 0
 neg_tau_constr_15_6: tau[15,6] + neg_tau_var_15_6 = 0
 tau_computation_15_7: - 0.1308042153486798 betas[7,0]
   - 0.4390934242476194 betas[7,1] + 0.224750820811735 betas[7,2]
   - 0.2051628180813711 betas[7,3] - betas[7,4] + tau[15,7] = 0
 neg_tau_constr_15_7: tau[15,7] + neg_tau_var_15_7 = 0
 sum_r_15: r[15,0] + r[15,1] + r[15,2] + r[15,3] + r[15,4] + r[15,5]
   + r[15,6] + r[15,7] = 1
 Ensure_only_one_R_true_15: R[15,0] + R[15,1] + R[15,2] + R[15,3] + R[15,4]
   + R[15,5] + R[15,6] + R[15,7] = 1
 Define_Q_hat_15: - 1919.807661232052 R[15,0] - 1904.320264092098 R[15,1]
   - 1907.61094671161 R[15,2] - 1920.636337981252 R[15,3]
   - 1929.459406664147 R[15,4] - 1942.304245639772 R[15,5]
   - 1942.966799692372 R[15,6] - 1940.322638030631 R[15,7] + Q_hat[15] = 0
 Adjust_Q_hat_15: Q0_var[15] - Q_hat[15] + Q_hat_adjusted[15] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_15: Total_Demand_Up_to_K_minus_1[15]
   - 193.0467463732862 R[15,0] - 383.4022039741387 R[15,1]
   - 574.6790308934252 R[15,2] - 767.601428857683 R[15,3]
   - 961.2739400410825 R[15,4] - 1160.521966082825 R[15,5]
   - 1358.021980948687 R[15,6] - 1552.668572630034 R[15,7] = 0
 Constr_Total_Demand_from_K_to_T_15: Total_Demand_from_k_to_T[15]
   - 1750.895415839532 R[15,0] - 1560.53995823868 R[15,1]
   - 1369.263131319393 R[15,2] - 1176.340733355135 R[15,3]
   - 982.6682221717357 R[15,4] - 783.4201961299927 R[15,5]
   - 585.9201812641307 R[15,6] - 391.2735895827847 R[15,7] = 0
 Constr_Left_0_diff_aux_15: Sold_0[15] - Q0_var[15] + Left_0_aux_15 = 0
 Constr_Lost_0_diff_aux_15: - Total_Demand_Up_to_K_minus_1[15] + Q0_var[15]
   + Lost_0_aux_15 = 0
 Constr_Q1_plus_left_15: - Left_0[15] + Q1_plus_left[15] - Q1_var[15] = 0
 Constr_Left_1_diff_aux_15: Sold_1[15] - Q1_plus_left[15] + Left_0_aux_15
   = 0
 Constr_Lost_1_diff_aux_15: - Total_Demand_from_k_to_T[15]
   + Q1_plus_left[15] + Lost_0_aux_15 = 0
 Profit_Constraint_15: - 500 Sold_0[15] + 500 Lost_0[15] - 500 Sold_1[15]
   + 500 Left_1[15] + 500 Lost_1[15] + profits_vars[15] = 0
 R480: - 0.8506580589289626 alphas[0] - 0.2486480699198531 alphas[1]
   + 0.5309397405277253 alphas[2] + 1.48725018766534 alphas[3] - alphas[4]
   + f_var[16] = 0
 Q0_upper_bound_16:
   - 1829.241320246946 Fraction_for_second_order_amount[16] + Q0_var[16]
   = 0
 tau_computation_16_0: - 0.8506580589289626 betas[0,0]
   - 0.2486480699198531 betas[0,1] + 0.5309397405277253 betas[0,2]
   + 1.48725018766534 betas[0,3] - betas[0,4] + tau[16,0] = 0
 neg_tau_constr_16_0: tau[16,0] + neg_tau_var_16_0 = 0
 tau_computation_16_1: - 0.8506580589289626 betas[1,0]
   - 0.2486480699198531 betas[1,1] + 0.5309397405277253 betas[1,2]
   + 1.48725018766534 betas[1,3] - betas[1,4] + tau[16,1] = 0
 neg_tau_constr_16_1: tau[16,1] + neg_tau_var_16_1 = 0
 tau_computation_16_2: - 0.8506580589289626 betas[2,0]
   - 0.2486480699198531 betas[2,1] + 0.5309397405277253 betas[2,2]
   + 1.48725018766534 betas[2,3] - betas[2,4] + tau[16,2] = 0
 neg_tau_constr_16_2: tau[16,2] + neg_tau_var_16_2 = 0
 tau_computation_16_3: - 0.8506580589289626 betas[3,0]
   - 0.2486480699198531 betas[3,1] + 0.5309397405277253 betas[3,2]
   + 1.48725018766534 betas[3,3] - betas[3,4] + tau[16,3] = 0
 neg_tau_constr_16_3: tau[16,3] + neg_tau_var_16_3 = 0
 tau_computation_16_4: - 0.8506580589289626 betas[4,0]
   - 0.2486480699198531 betas[4,1] + 0.5309397405277253 betas[4,2]
   + 1.48725018766534 betas[4,3] - betas[4,4] + tau[16,4] = 0
 neg_tau_constr_16_4: tau[16,4] + neg_tau_var_16_4 = 0
 tau_computation_16_5: - 0.8506580589289626 betas[5,0]
   - 0.2486480699198531 betas[5,1] + 0.5309397405277253 betas[5,2]
   + 1.48725018766534 betas[5,3] - betas[5,4] + tau[16,5] = 0
 neg_tau_constr_16_5: tau[16,5] + neg_tau_var_16_5 = 0
 tau_computation_16_6: - 0.8506580589289626 betas[6,0]
   - 0.2486480699198531 betas[6,1] + 0.5309397405277253 betas[6,2]
   + 1.48725018766534 betas[6,3] - betas[6,4] + tau[16,6] = 0
 neg_tau_constr_16_6: tau[16,6] + neg_tau_var_16_6 = 0
 tau_computation_16_7: - 0.8506580589289626 betas[7,0]
   - 0.2486480699198531 betas[7,1] + 0.5309397405277253 betas[7,2]
   + 1.48725018766534 betas[7,3] - betas[7,4] + tau[16,7] = 0
 neg_tau_constr_16_7: tau[16,7] + neg_tau_var_16_7 = 0
 sum_r_16: r[16,0] + r[16,1] + r[16,2] + r[16,3] + r[16,4] + r[16,5]
   + r[16,6] + r[16,7] = 1
 Ensure_only_one_R_true_16: R[16,0] + R[16,1] + R[16,2] + R[16,3] + R[16,4]
   + R[16,5] + R[16,6] + R[16,7] = 1
 Define_Q_hat_16: - 1724.948193481366 R[16,0] - 1757.179887438736 R[16,1]
   - 1757.533574088015 R[16,2] - 1766.185798370599 R[16,3]
   - 1785.289990333466 R[16,4] - 1795.859765007269 R[16,5]
   - 1800.327882016599 R[16,6] - 1797.727727191046 R[16,7] + Q_hat[16] = 0
 Adjust_Q_hat_16: Q0_var[16] - Q_hat[16] + Q_hat_adjusted[16] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_16: Total_Demand_Up_to_K_minus_1[16]
   - 173.9616869656109 R[16,0] - 352.5032717312452 R[16,1]
   - 528.7079057675027 R[16,2] - 705.4657055431499 R[16,3]
   - 884.4481492177032 R[16,4] - 1070.463687338341 R[16,5]
   - 1253.842158796782 R[16,6] - 1437.507398928891 R[16,7] = 0
 Constr_Total_Demand_from_K_to_T_16: Total_Demand_from_k_to_T[16]
   - 1620.888802481792 R[16,0] - 1442.347217716158 R[16,1]
   - 1266.142583679901 R[16,2] - 1089.384783904253 R[16,3]
   - 910.4023402297003 R[16,4] - 724.3868021090625 R[16,5]
   - 541.0083306506214 R[16,6] - 357.3430905185121 R[16,7] = 0
 Constr_Left_0_diff_aux_16: Sold_0[16] - Q0_var[16] + Left_0_aux_16 = 0
 Constr_Lost_0_diff_aux_16: - Total_Demand_Up_to_K_minus_1[16] + Q0_var[16]
   + Lost_0_aux_16 = 0
 Constr_Q1_plus_left_16: - Left_0[16] + Q1_plus_left[16] - Q1_var[16] = 0
 Constr_Left_1_diff_aux_16: Sold_1[16] - Q1_plus_left[16] + Left_0_aux_16
   = 0
 Constr_Lost_1_diff_aux_16: - Total_Demand_from_k_to_T[16]
   + Q1_plus_left[16] + Lost_0_aux_16 = 0
 Profit_Constraint_16: - 500 Sold_0[16] + 500 Lost_0[16] - 500 Sold_1[16]
   + 500 Left_1[16] + 500 Lost_1[16] + profits_vars[16] = 0
 R510: - 1.123316297134206 alphas[0] + 1.050894787376198 alphas[1]
   - 0.2710442765598579 alphas[2] + 1.074808636788597 alphas[3] - alphas[4]
   + f_var[17] = 0
 Q0_upper_bound_17:
   - 1829.241320246946 Fraction_for_second_order_amount[17] + Q0_var[17]
   = 0
 tau_computation_17_0: - 1.123316297134206 betas[0,0]
   + 1.050894787376198 betas[0,1] - 0.2710442765598579 betas[0,2]
   + 1.074808636788597 betas[0,3] - betas[0,4] + tau[17,0] = 0
 neg_tau_constr_17_0: tau[17,0] + neg_tau_var_17_0 = 0
 tau_computation_17_1: - 1.123316297134206 betas[1,0]
   + 1.050894787376198 betas[1,1] - 0.2710442765598579 betas[1,2]
   + 1.074808636788597 betas[1,3] - betas[1,4] + tau[17,1] = 0
 neg_tau_constr_17_1: tau[17,1] + neg_tau_var_17_1 = 0
 tau_computation_17_2: - 1.123316297134206 betas[2,0]
   + 1.050894787376198 betas[2,1] - 0.2710442765598579 betas[2,2]
   + 1.074808636788597 betas[2,3] - betas[2,4] + tau[17,2] = 0
 neg_tau_constr_17_2: tau[17,2] + neg_tau_var_17_2 = 0
 tau_computation_17_3: - 1.123316297134206 betas[3,0]
   + 1.050894787376198 betas[3,1] - 0.2710442765598579 betas[3,2]
   + 1.074808636788597 betas[3,3] - betas[3,4] + tau[17,3] = 0
 neg_tau_constr_17_3: tau[17,3] + neg_tau_var_17_3 = 0
 tau_computation_17_4: - 1.123316297134206 betas[4,0]
   + 1.050894787376198 betas[4,1] - 0.2710442765598579 betas[4,2]
   + 1.074808636788597 betas[4,3] - betas[4,4] + tau[17,4] = 0
 neg_tau_constr_17_4: tau[17,4] + neg_tau_var_17_4 = 0
 tau_computation_17_5: - 1.123316297134206 betas[5,0]
   + 1.050894787376198 betas[5,1] - 0.2710442765598579 betas[5,2]
   + 1.074808636788597 betas[5,3] - betas[5,4] + tau[17,5] = 0
 neg_tau_constr_17_5: tau[17,5] + neg_tau_var_17_5 = 0
 tau_computation_17_6: - 1.123316297134206 betas[6,0]
   + 1.050894787376198 betas[6,1] - 0.2710442765598579 betas[6,2]
   + 1.074808636788597 betas[6,3] - betas[6,4] + tau[17,6] = 0
 neg_tau_constr_17_6: tau[17,6] + neg_tau_var_17_6 = 0
 tau_computation_17_7: - 1.123316297134206 betas[7,0]
   + 1.050894787376198 betas[7,1] - 0.2710442765598579 betas[7,2]
   + 1.074808636788597 betas[7,3] - betas[7,4] + tau[17,7] = 0
 neg_tau_constr_17_7: tau[17,7] + neg_tau_var_17_7 = 0
 sum_r_17: r[17,0] + r[17,1] + r[17,2] + r[17,3] + r[17,4] + r[17,5]
   + r[17,6] + r[17,7] = 1
 Ensure_only_one_R_true_17: R[17,0] + R[17,1] + R[17,2] + R[17,3] + R[17,4]
   + R[17,5] + R[17,6] + R[17,7] = 1
 Define_Q_hat_17: - 1761.542352631085 R[17,0] - 1767.04798932546 R[17,1]
   - 1764.795887146353 R[17,2] - 1683.751124086871 R[17,3]
   - 1689.439322154392 R[17,4] - 1688.206929905802 R[17,5]
   - 1692.660010985329 R[17,6] - 1682.437185448538 R[17,7] + Q_hat[17] = 0
 Adjust_Q_hat_17: Q0_var[17] - Q_hat[17] + Q_hat_adjusted[17] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_17: Total_Demand_Up_to_K_minus_1[17]
   - 177.5458171565026 R[17,0] - 355.0892929505559 R[17,1]
   - 530.8809114345012 R[17,2] - 697.6178279489491 R[17,3]
   - 860.8880720736825 R[17,4] - 1022.652795210262 R[17,5]
   - 1184.026923047461 R[17,6] - 1355.24993955765 R[17,7] = 0
 Constr_Total_Demand_from_K_to_T_17: Total_Demand_from_k_to_T[17]
   - 1511.352924222126 R[17,0] - 1333.809448428073 R[17,1]
   - 1158.017829944128 R[17,2] - 991.2809134296799 R[17,3]
   - 828.0106693049464 R[17,4] - 666.2459461683675 R[17,5]
   - 504.8718183311677 R[17,6] - 333.6488018209786 R[17,7] = 0
 Constr_Left_0_diff_aux_17: Sold_0[17] - Q0_var[17] + Left_0_aux_17 = 0
 Constr_Lost_0_diff_aux_17: - Total_Demand_Up_to_K_minus_1[17] + Q0_var[17]
   + Lost_0_aux_17 = 0
 Constr_Q1_plus_left_17: - Left_0[17] + Q1_plus_left[17] - Q1_var[17] = 0
 Constr_Left_1_diff_aux_17: Sold_1[17] - Q1_plus_left[17] + Left_0_aux_17
   = 0
 Constr_Lost_1_diff_aux_17: - Total_Demand_from_k_to_T[17]
   + Q1_plus_left[17] + Lost_0_aux_17 = 0
 Profit_Constraint_17: - 500 Sold_0[17] + 500 Lost_0[17] - 500 Sold_1[17]
   + 500 Left_1[17] + 500 Lost_1[17] + profits_vars[17] = 0
 R540: 0.7491012015205633 alphas[0] + 0.5334982307492885 alphas[1]
   - 1.362479139188685 alphas[2] + 1.435105839254642 alphas[3] - alphas[4]
   + f_var[18] = 0
 Q0_upper_bound_18:
   - 1829.241320246946 Fraction_for_second_order_amount[18] + Q0_var[18]
   = 0
 tau_computation_18_0: 0.7491012015205633 betas[0,0]
   + 0.5334982307492885 betas[0,1] - 1.362479139188685 betas[0,2]
   + 1.435105839254642 betas[0,3] - betas[0,4] + tau[18,0] = 0
 neg_tau_constr_18_0: tau[18,0] + neg_tau_var_18_0 = 0
 tau_computation_18_1: 0.7491012015205633 betas[1,0]
   + 0.5334982307492885 betas[1,1] - 1.362479139188685 betas[1,2]
   + 1.435105839254642 betas[1,3] - betas[1,4] + tau[18,1] = 0
 neg_tau_constr_18_1: tau[18,1] + neg_tau_var_18_1 = 0
 tau_computation_18_2: 0.7491012015205633 betas[2,0]
   + 0.5334982307492885 betas[2,1] - 1.362479139188685 betas[2,2]
   + 1.435105839254642 betas[2,3] - betas[2,4] + tau[18,2] = 0
 neg_tau_constr_18_2: tau[18,2] + neg_tau_var_18_2 = 0
 tau_computation_18_3: 0.7491012015205633 betas[3,0]
   + 0.5334982307492885 betas[3,1] - 1.362479139188685 betas[3,2]
   + 1.435105839254642 betas[3,3] - betas[3,4] + tau[18,3] = 0
 neg_tau_constr_18_3: tau[18,3] + neg_tau_var_18_3 = 0
 tau_computation_18_4: 0.7491012015205633 betas[4,0]
   + 0.5334982307492885 betas[4,1] - 1.362479139188685 betas[4,2]
   + 1.435105839254642 betas[4,3] - betas[4,4] + tau[18,4] = 0
 neg_tau_constr_18_4: tau[18,4] + neg_tau_var_18_4 = 0
 tau_computation_18_5: 0.7491012015205633 betas[5,0]
   + 0.5334982307492885 betas[5,1] - 1.362479139188685 betas[5,2]
   + 1.435105839254642 betas[5,3] - betas[5,4] + tau[18,5] = 0
 neg_tau_constr_18_5: tau[18,5] + neg_tau_var_18_5 = 0
 tau_computation_18_6: 0.7491012015205633 betas[6,0]
   + 0.5334982307492885 betas[6,1] - 1.362479139188685 betas[6,2]
   + 1.435105839254642 betas[6,3] - betas[6,4] + tau[18,6] = 0
 neg_tau_constr_18_6: tau[18,6] + neg_tau_var_18_6 = 0
 tau_computation_18_7: 0.7491012015205633 betas[7,0]
   + 0.5334982307492885 betas[7,1] - 1.362479139188685 betas[7,2]
   + 1.435105839254642 betas[7,3] - betas[7,4] + tau[18,7] = 0
 neg_tau_constr_18_7: tau[18,7] + neg_tau_var_18_7 = 0
 sum_r_18: r[18,0] + r[18,1] + r[18,2] + r[18,3] + r[18,4] + r[18,5]
   + r[18,6] + r[18,7] = 1
 Ensure_only_one_R_true_18: R[18,0] + R[18,1] + R[18,2] + R[18,3] + R[18,4]
   + R[18,5] + R[18,6] + R[18,7] = 1
 Define_Q_hat_18: - 1515.905989047141 R[18,0] - 1469.446366488893 R[18,1]
   - 1471.526491351736 R[18,2] - 1461.334482876932 R[18,3]
   - 1470.959248842689 R[18,4] - 1477.894058343894 R[18,5]
   - 1461.380388537297 R[18,6] - 1452.957921204186 R[18,7] + Q_hat[18] = 0
 Adjust_Q_hat_18: Q0_var[18] - Q_hat[18] + Q_hat_adjusted[18] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_18: Total_Demand_Up_to_K_minus_1[18]
   - 153.4875322509779 R[18,0] - 295.8365835577187 R[18,1]
   - 443.4899251664913 R[18,2] - 591.9218554429332 R[18,3]
   - 737.8159514887868 R[18,4] - 886.1855812061245 R[18,5]
   - 1029.419568088453 R[18,6] - 1172.052768299522 R[18,7] = 0
 Constr_Total_Demand_from_K_to_T_18: Total_Demand_from_k_to_T[18]
   - 1295.707768965765 R[18,0] - 1153.358717659024 R[18,1]
   - 1005.705376050251 R[18,2] - 857.2734457738093 R[18,3]
   - 711.3793497279556 R[18,4] - 563.009720010618 R[18,5]
   - 419.7757331282897 R[18,6] - 277.1425329172206 R[18,7] = 0
 Constr_Left_0_diff_aux_18: Sold_0[18] - Q0_var[18] + Left_0_aux_18 = 0
 Constr_Lost_0_diff_aux_18: - Total_Demand_Up_to_K_minus_1[18] + Q0_var[18]
   + Lost_0_aux_18 = 0
 Constr_Q1_plus_left_18: - Left_0[18] + Q1_plus_left[18] - Q1_var[18] = 0
 Constr_Left_1_diff_aux_18: Sold_1[18] - Q1_plus_left[18] + Left_0_aux_18
   = 0
 Constr_Lost_1_diff_aux_18: - Total_Demand_from_k_to_T[18]
   + Q1_plus_left[18] + Lost_0_aux_18 = 0
 Profit_Constraint_18: - 500 Sold_0[18] + 500 Lost_0[18] - 500 Sold_1[18]
   + 500 Left_1[18] + 500 Lost_1[18] + profits_vars[18] = 0
 R570: - 0.3991178146662819 alphas[0] - 1.855475140457852 alphas[1]
   + 0.8241475768256888 alphas[2] + 0.5814073737540502 alphas[3]
   - alphas[4] + f_var[19] = 0
 Q0_upper_bound_19:
   - 1829.241320246946 Fraction_for_second_order_amount[19] + Q0_var[19]
   = 0
 tau_computation_19_0: - 0.3991178146662819 betas[0,0]
   - 1.855475140457852 betas[0,1] + 0.8241475768256888 betas[0,2]
   + 0.5814073737540502 betas[0,3] - betas[0,4] + tau[19,0] = 0
 neg_tau_constr_19_0: tau[19,0] + neg_tau_var_19_0 = 0
 tau_computation_19_1: - 0.3991178146662819 betas[1,0]
   - 1.855475140457852 betas[1,1] + 0.8241475768256888 betas[1,2]
   + 0.5814073737540502 betas[1,3] - betas[1,4] + tau[19,1] = 0
 neg_tau_constr_19_1: tau[19,1] + neg_tau_var_19_1 = 0
 tau_computation_19_2: - 0.3991178146662819 betas[2,0]
   - 1.855475140457852 betas[2,1] + 0.8241475768256888 betas[2,2]
   + 0.5814073737540502 betas[2,3] - betas[2,4] + tau[19,2] = 0
 neg_tau_constr_19_2: tau[19,2] + neg_tau_var_19_2 = 0
 tau_computation_19_3: - 0.3991178146662819 betas[3,0]
   - 1.855475140457852 betas[3,1] + 0.8241475768256888 betas[3,2]
   + 0.5814073737540502 betas[3,3] - betas[3,4] + tau[19,3] = 0
 neg_tau_constr_19_3: tau[19,3] + neg_tau_var_19_3 = 0
 tau_computation_19_4: - 0.3991178146662819 betas[4,0]
   - 1.855475140457852 betas[4,1] + 0.8241475768256888 betas[4,2]
   + 0.5814073737540502 betas[4,3] - betas[4,4] + tau[19,4] = 0
 neg_tau_constr_19_4: tau[19,4] + neg_tau_var_19_4 = 0
 tau_computation_19_5: - 0.3991178146662819 betas[5,0]
   - 1.855475140457852 betas[5,1] + 0.8241475768256888 betas[5,2]
   + 0.5814073737540502 betas[5,3] - betas[5,4] + tau[19,5] = 0
 neg_tau_constr_19_5: tau[19,5] + neg_tau_var_19_5 = 0
 tau_computation_19_6: - 0.3991178146662819 betas[6,0]
   - 1.855475140457852 betas[6,1] + 0.8241475768256888 betas[6,2]
   + 0.5814073737540502 betas[6,3] - betas[6,4] + tau[19,6] = 0
 neg_tau_constr_19_6: tau[19,6] + neg_tau_var_19_6 = 0
 tau_computation_19_7: - 0.3991178146662819 betas[7,0]
   - 1.855475140457852 betas[7,1] + 0.8241475768256888 betas[7,2]
   + 0.5814073737540502 betas[7,3] - betas[7,4] + tau[19,7] = 0
 neg_tau_constr_19_7: tau[19,7] + neg_tau_var_19_7 = 0
 sum_r_19: r[19,0] + r[19,1] + r[19,2] + r[19,3] + r[19,4] + r[19,5]
   + r[19,6] + r[19,7] = 1
 Ensure_only_one_R_true_19: R[19,0] + R[19,1] + R[19,2] + R[19,3] + R[19,4]
   + R[19,5] + R[19,6] + R[19,7] = 1
 Define_Q_hat_19: - 1959.724718446498 R[19,0] - 2013.20002312549 R[19,1]
   - 2013.009343822453 R[19,2] - 2006.381180912848 R[19,3]
   - 2039.26374288758 R[19,4] - 2031.411105854591 R[19,5]
   - 2035.759359521818 R[19,6] - 2042.012397854521 R[19,7] + Q_hat[19] = 0
 Adjust_Q_hat_19: Q0_var[19] - Q_hat[19] + Q_hat_adjusted[19] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_19: Total_Demand_Up_to_K_minus_1[19]
   - 196.9563301215698 R[19,0] - 403.989417382072 R[19,1]
   - 605.4035453413539 R[19,2] - 804.4935832981573 R[19,3]
   - 1011.964964776724 R[19,4] - 1214.518758536527 R[19,5]
   - 1416.912686429062 R[19,6] - 1629.556004938272 R[19,7] = 0
 Constr_Total_Demand_from_K_to_T_19: Total_Demand_from_k_to_T[19]
   - 1841.115768904788 R[19,0] - 1634.082681644286 R[19,1]
   - 1432.668553685004 R[19,2] - 1233.578515728201 R[19,3]
   - 1026.107134249634 R[19,4] - 823.5533404898314 R[19,5]
   - 621.1594125972958 R[19,6] - 408.5160940880857 R[19,7] = 0
 Constr_Left_0_diff_aux_19: Sold_0[19] - Q0_var[19] + Left_0_aux_19 = 0
 Constr_Lost_0_diff_aux_19: - Total_Demand_Up_to_K_minus_1[19] + Q0_var[19]
   + Lost_0_aux_19 = 0
 Constr_Q1_plus_left_19: - Left_0[19] + Q1_plus_left[19] - Q1_var[19] = 0
 Constr_Left_1_diff_aux_19: Sold_1[19] - Q1_plus_left[19] + Left_0_aux_19
   = 0
 Constr_Lost_1_diff_aux_19: - Total_Demand_from_k_to_T[19]
   + Q1_plus_left[19] + Lost_0_aux_19 = 0
 Profit_Constraint_19: - 500 Sold_0[19] + 500 Lost_0[19] - 500 Sold_1[19]
   + 500 Left_1[19] + 500 Lost_1[19] + profits_vars[19] = 0
 R600: 1.415065403221863 alphas[0] - 0.6597746969314081 alphas[1]
   + 2.213270564167405 alphas[2] + 0.8786478363208375 alphas[3] - alphas[4]
   + f_var[20] = 0
 Q0_upper_bound_20:
   - 1829.241320246946 Fraction_for_second_order_amount[20] + Q0_var[20]
   = 0
 tau_computation_20_0: 1.415065403221863 betas[0,0]
   - 0.6597746969314081 betas[0,1] + 2.213270564167405 betas[0,2]
   + 0.8786478363208375 betas[0,3] - betas[0,4] + tau[20,0] = 0
 neg_tau_constr_20_0: tau[20,0] + neg_tau_var_20_0 = 0
 tau_computation_20_1: 1.415065403221863 betas[1,0]
   - 0.6597746969314081 betas[1,1] + 2.213270564167405 betas[1,2]
   + 0.8786478363208375 betas[1,3] - betas[1,4] + tau[20,1] = 0
 neg_tau_constr_20_1: tau[20,1] + neg_tau_var_20_1 = 0
 tau_computation_20_2: 1.415065403221863 betas[2,0]
   - 0.6597746969314081 betas[2,1] + 2.213270564167405 betas[2,2]
   + 0.8786478363208375 betas[2,3] - betas[2,4] + tau[20,2] = 0
 neg_tau_constr_20_2: tau[20,2] + neg_tau_var_20_2 = 0
 tau_computation_20_3: 1.415065403221863 betas[3,0]
   - 0.6597746969314081 betas[3,1] + 2.213270564167405 betas[3,2]
   + 0.8786478363208375 betas[3,3] - betas[3,4] + tau[20,3] = 0
 neg_tau_constr_20_3: tau[20,3] + neg_tau_var_20_3 = 0
 tau_computation_20_4: 1.415065403221863 betas[4,0]
   - 0.6597746969314081 betas[4,1] + 2.213270564167405 betas[4,2]
   + 0.8786478363208375 betas[4,3] - betas[4,4] + tau[20,4] = 0
 neg_tau_constr_20_4: tau[20,4] + neg_tau_var_20_4 = 0
 tau_computation_20_5: 1.415065403221863 betas[5,0]
   - 0.6597746969314081 betas[5,1] + 2.213270564167405 betas[5,2]
   + 0.8786478363208375 betas[5,3] - betas[5,4] + tau[20,5] = 0
 neg_tau_constr_20_5: tau[20,5] + neg_tau_var_20_5 = 0
 tau_computation_20_6: 1.415065403221863 betas[6,0]
   - 0.6597746969314081 betas[6,1] + 2.213270564167405 betas[6,2]
   + 0.8786478363208375 betas[6,3] - betas[6,4] + tau[20,6] = 0
 neg_tau_constr_20_6: tau[20,6] + neg_tau_var_20_6 = 0
 tau_computation_20_7: 1.415065403221863 betas[7,0]
   - 0.6597746969314081 betas[7,1] + 2.213270564167405 betas[7,2]
   + 0.8786478363208375 betas[7,3] - betas[7,4] + tau[20,7] = 0
 neg_tau_constr_20_7: tau[20,7] + neg_tau_var_20_7 = 0
 sum_r_20: r[20,0] + r[20,1] + r[20,2] + r[20,3] + r[20,4] + r[20,5]
   + r[20,6] + r[20,7] = 1
 Ensure_only_one_R_true_20: R[20,0] + R[20,1] + R[20,2] + R[20,3] + R[20,4]
   + R[20,5] + R[20,6] + R[20,7] = 1
 Define_Q_hat_20: - 1579.056075040951 R[20,0] - 1601.756535073794 R[20,1]
   - 1607.988535910145 R[20,2] - 1654.170390656788 R[20,3]
   - 1618.211663087145 R[20,4] - 1621.420217159943 R[20,5]
   - 1624.221000256109 R[20,6] - 1627.457681554218 R[20,7] + Q_hat[20] = 0
 Adjust_Q_hat_20: Q0_var[20] - Q_hat[20] + Q_hat_adjusted[20] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_20: Total_Demand_Up_to_K_minus_1[20]
   - 159.6726212112623 R[20,0] - 321.1738442352167 R[20,1]
   - 484.3641168825873 R[20,2] - 651.0418660276755 R[20,3]
   - 802.5131594011145 R[20,4] - 964.8157856117361 R[20,5]
   - 1128.342045842597 R[20,6] - 1287.728823133599 R[20,7] = 0
 Constr_Total_Demand_from_K_to_T_20: Total_Demand_from_k_to_T[20]
   - 1461.364297594603 R[20,0] - 1299.863074570648 R[20,1]
   - 1136.672801923278 R[20,2] - 969.9950527781896 R[20,3]
   - 818.5237594047505 R[20,4] - 656.2211331941289 R[20,5]
   - 492.694872963268 R[20,6] - 333.3080956722663 R[20,7] = 0
 Constr_Left_0_diff_aux_20: Sold_0[20] - Q0_var[20] + Left_0_aux_20 = 0
 Constr_Lost_0_diff_aux_20: - Total_Demand_Up_to_K_minus_1[20] + Q0_var[20]
   + Lost_0_aux_20 = 0
 Constr_Q1_plus_left_20: - Left_0[20] + Q1_plus_left[20] - Q1_var[20] = 0
 Constr_Left_1_diff_aux_20: Sold_1[20] - Q1_plus_left[20] + Left_0_aux_20
   = 0
 Constr_Lost_1_diff_aux_20: - Total_Demand_from_k_to_T[20]
   + Q1_plus_left[20] + Lost_0_aux_20 = 0
 Profit_Constraint_20: - 500 Sold_0[20] + 500 Lost_0[20] - 500 Sold_1[20]
   + 500 Left_1[20] + 500 Lost_1[20] + profits_vars[20] = 0
 R630: 0.1278425830839482 alphas[0] + 1.030152477869554 alphas[1]
   - 1.40207155387602 alphas[2] - 0.486215275191437 alphas[3] - alphas[4]
   + f_var[21] = 0
 Q0_upper_bound_21:
   - 1829.241320246946 Fraction_for_second_order_amount[21] + Q0_var[21]
   = 0
 tau_computation_21_0: 0.1278425830839482 betas[0,0]
   + 1.030152477869554 betas[0,1] - 1.40207155387602 betas[0,2]
   - 0.486215275191437 betas[0,3] - betas[0,4] + tau[21,0] = 0
 neg_tau_constr_21_0: tau[21,0] + neg_tau_var_21_0 = 0
 tau_computation_21_1: 0.1278425830839482 betas[1,0]
   + 1.030152477869554 betas[1,1] - 1.40207155387602 betas[1,2]
   - 0.486215275191437 betas[1,3] - betas[1,4] + tau[21,1] = 0
 neg_tau_constr_21_1: tau[21,1] + neg_tau_var_21_1 = 0
 tau_computation_21_2: 0.1278425830839482 betas[2,0]
   + 1.030152477869554 betas[2,1] - 1.40207155387602 betas[2,2]
   - 0.486215275191437 betas[2,3] - betas[2,4] + tau[21,2] = 0
 neg_tau_constr_21_2: tau[21,2] + neg_tau_var_21_2 = 0
 tau_computation_21_3: 0.1278425830839482 betas[3,0]
   + 1.030152477869554 betas[3,1] - 1.40207155387602 betas[3,2]
   - 0.486215275191437 betas[3,3] - betas[3,4] + tau[21,3] = 0
 neg_tau_constr_21_3: tau[21,3] + neg_tau_var_21_3 = 0
 tau_computation_21_4: 0.1278425830839482 betas[4,0]
   + 1.030152477869554 betas[4,1] - 1.40207155387602 betas[4,2]
   - 0.486215275191437 betas[4,3] - betas[4,4] + tau[21,4] = 0
 neg_tau_constr_21_4: tau[21,4] + neg_tau_var_21_4 = 0
 tau_computation_21_5: 0.1278425830839482 betas[5,0]
   + 1.030152477869554 betas[5,1] - 1.40207155387602 betas[5,2]
   - 0.486215275191437 betas[5,3] - betas[5,4] + tau[21,5] = 0
 neg_tau_constr_21_5: tau[21,5] + neg_tau_var_21_5 = 0
 tau_computation_21_6: 0.1278425830839482 betas[6,0]
   + 1.030152477869554 betas[6,1] - 1.40207155387602 betas[6,2]
   - 0.486215275191437 betas[6,3] - betas[6,4] + tau[21,6] = 0
 neg_tau_constr_21_6: tau[21,6] + neg_tau_var_21_6 = 0
 tau_computation_21_7: 0.1278425830839482 betas[7,0]
   + 1.030152477869554 betas[7,1] - 1.40207155387602 betas[7,2]
   - 0.486215275191437 betas[7,3] - betas[7,4] + tau[21,7] = 0
 neg_tau_constr_21_7: tau[21,7] + neg_tau_var_21_7 = 0
 sum_r_21: r[21,0] + r[21,1] + r[21,2] + r[21,3] + r[21,4] + r[21,5]
   + r[21,6] + r[21,7] = 1
 Ensure_only_one_R_true_21: R[21,0] + R[21,1] + R[21,2] + R[21,3] + R[21,4]
   + R[21,5] + R[21,6] + R[21,7] = 1
 Define_Q_hat_21: - 1667.811638624297 R[21,0] - 1733.714542742281 R[21,1]
   - 1731.560931968201 R[21,2] - 1678.666010567465 R[21,3]
   - 1712.568262578317 R[21,4] - 1710.942301466359 R[21,5]
   - 1715.561100087627 R[21,6] - 1717.668503598654 R[21,7] + Q_hat[21] = 0
 Adjust_Q_hat_21: Q0_var[21] - Q_hat[21] + Q_hat_adjusted[21] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_21: Total_Demand_Up_to_K_minus_1[21]
   - 168.3655793892921 R[21,0] - 347.0065111386832 R[21,1]
   - 520.3000697573514 R[21,2] - 685.7953738477603 R[21,3]
   - 857.2154496869376 R[21,4] - 1027.945315735737 R[21,5]
   - 1194.924604172265 R[21,6] - 1379.178835508619 R[21,7] = 0
 Constr_Total_Demand_from_K_to_T_21: Total_Demand_from_k_to_T[21]
   - 1547.76142660366 R[21,0] - 1369.120494854269 R[21,1]
   - 1195.826936235601 R[21,2] - 1030.331632145192 R[21,3]
   - 858.9115563060145 R[21,4] - 688.1816902572152 R[21,5]
   - 521.2024018206873 R[21,6] - 336.9481704843326 R[21,7] = 0
 Constr_Left_0_diff_aux_21: Sold_0[21] - Q0_var[21] + Left_0_aux_21 = 0
 Constr_Lost_0_diff_aux_21: - Total_Demand_Up_to_K_minus_1[21] + Q0_var[21]
   + Lost_0_aux_21 = 0
 Constr_Q1_plus_left_21: - Left_0[21] + Q1_plus_left[21] - Q1_var[21] = 0
 Constr_Left_1_diff_aux_21: Sold_1[21] - Q1_plus_left[21] + Left_0_aux_21
   = 0
 Constr_Lost_1_diff_aux_21: - Total_Demand_from_k_to_T[21]
   + Q1_plus_left[21] + Lost_0_aux_21 = 0
 Profit_Constraint_21: - 500 Sold_0[21] + 500 Lost_0[21] - 500 Sold_1[21]
   + 500 Left_1[21] + 500 Lost_1[21] + profits_vars[21] = 0
 R660: 0.1127179511384918 alphas[0] - 0.6028604682640695 alphas[1]
   - 0.209324649687527 alphas[2] + 1.152806658653639 alphas[3] - alphas[4]
   + f_var[22] = 0
 Q0_upper_bound_22:
   - 1829.241320246946 Fraction_for_second_order_amount[22] + Q0_var[22]
   = 0
 tau_computation_22_0: 0.1127179511384918 betas[0,0]
   - 0.6028604682640695 betas[0,1] - 0.209324649687527 betas[0,2]
   + 1.152806658653639 betas[0,3] - betas[0,4] + tau[22,0] = 0
 neg_tau_constr_22_0: tau[22,0] + neg_tau_var_22_0 = 0
 tau_computation_22_1: 0.1127179511384918 betas[1,0]
   - 0.6028604682640695 betas[1,1] - 0.209324649687527 betas[1,2]
   + 1.152806658653639 betas[1,3] - betas[1,4] + tau[22,1] = 0
 neg_tau_constr_22_1: tau[22,1] + neg_tau_var_22_1 = 0
 tau_computation_22_2: 0.1127179511384918 betas[2,0]
   - 0.6028604682640695 betas[2,1] - 0.209324649687527 betas[2,2]
   + 1.152806658653639 betas[2,3] - betas[2,4] + tau[22,2] = 0
 neg_tau_constr_22_2: tau[22,2] + neg_tau_var_22_2 = 0
 tau_computation_22_3: 0.1127179511384918 betas[3,0]
   - 0.6028604682640695 betas[3,1] - 0.209324649687527 betas[3,2]
   + 1.152806658653639 betas[3,3] - betas[3,4] + tau[22,3] = 0
 neg_tau_constr_22_3: tau[22,3] + neg_tau_var_22_3 = 0
 tau_computation_22_4: 0.1127179511384918 betas[4,0]
   - 0.6028604682640695 betas[4,1] - 0.209324649687527 betas[4,2]
   + 1.152806658653639 betas[4,3] - betas[4,4] + tau[22,4] = 0
 neg_tau_constr_22_4: tau[22,4] + neg_tau_var_22_4 = 0
 tau_computation_22_5: 0.1127179511384918 betas[5,0]
   - 0.6028604682640695 betas[5,1] - 0.209324649687527 betas[5,2]
   + 1.152806658653639 betas[5,3] - betas[5,4] + tau[22,5] = 0
 neg_tau_constr_22_5: tau[22,5] + neg_tau_var_22_5 = 0
 tau_computation_22_6: 0.1127179511384918 betas[6,0]
   - 0.6028604682640695 betas[6,1] - 0.209324649687527 betas[6,2]
   + 1.152806658653639 betas[6,3] - betas[6,4] + tau[22,6] = 0
 neg_tau_constr_22_6: tau[22,6] + neg_tau_var_22_6 = 0
 tau_computation_22_7: 0.1127179511384918 betas[7,0]
   - 0.6028604682640695 betas[7,1] - 0.209324649687527 betas[7,2]
   + 1.152806658653639 betas[7,3] - betas[7,4] + tau[22,7] = 0
 neg_tau_constr_22_7: tau[22,7] + neg_tau_var_22_7 = 0
 sum_r_22: r[22,0] + r[22,1] + r[22,2] + r[22,3] + r[22,4] + r[22,5]
   + r[22,6] + r[22,7] = 1
 Ensure_only_one_R_true_22: R[22,0] + R[22,1] + R[22,2] + R[22,3] + R[22,4]
   + R[22,5] + R[22,6] + R[22,7] = 1
 Define_Q_hat_22: - 1744.681853173633 R[22,0] - 1651.110984468502 R[22,1]
   - 1646.292923061378 R[22,2] - 1669.365589268106 R[22,3]
   - 1733.551220522965 R[22,4] - 1735.40069732284 R[22,5]
   - 1735.965788531209 R[22,6] - 1741.409941754659 R[22,7] + Q_hat[22] = 0
 Adjust_Q_hat_22: Q0_var[22] - Q_hat[22] + Q_hat_adjusted[22] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_22: Total_Demand_Up_to_K_minus_1[22]
   - 175.8944545745198 R[22,0] - 333.7276928991292 R[22,1]
   - 495.8687229665588 R[22,2] - 667.8679473263614 R[22,3]
   - 854.7646300390506 R[22,4] - 1032.917682880886 R[22,5]
   - 1216.470593117662 R[22,6] - 1389.961608116605 R[22,7] = 0
 Constr_Total_Demand_from_K_to_T_22: Total_Demand_from_k_to_T[22]
   - 1564.326474194301 R[22,0] - 1406.493235869691 R[22,1]
   - 1244.352205802262 R[22,2] - 1072.352981442459 R[22,3]
   - 885.4562987297699 R[22,4] - 707.3032458879345 R[22,5]
   - 523.7503356511583 R[22,6] - 350.259320652215 R[22,7] = 0
 Constr_Left_0_diff_aux_22: Sold_0[22] - Q0_var[22] + Left_0_aux_22 = 0
 Constr_Lost_0_diff_aux_22: - Total_Demand_Up_to_K_minus_1[22] + Q0_var[22]
   + Lost_0_aux_22 = 0
 Constr_Q1_plus_left_22: - Left_0[22] + Q1_plus_left[22] - Q1_var[22] = 0
 Constr_Left_1_diff_aux_22: Sold_1[22] - Q1_plus_left[22] + Left_0_aux_22
   = 0
 Constr_Lost_1_diff_aux_22: - Total_Demand_from_k_to_T[22]
   + Q1_plus_left[22] + Lost_0_aux_22 = 0
 Profit_Constraint_22: - 500 Sold_0[22] + 500 Lost_0[22] - 500 Sold_1[22]
   + 500 Left_1[22] + 500 Lost_1[22] + profits_vars[22] = 0
 R690: 1.72852479386002 alphas[0] + 1.315966968302297 alphas[1]
   - 1.222949256516093 alphas[2] + 0.7839101778757477 alphas[3] - alphas[4]
   + f_var[23] = 0
 Q0_upper_bound_23:
   - 1829.241320246946 Fraction_for_second_order_amount[23] + Q0_var[23]
   = 0
 tau_computation_23_0: 1.72852479386002 betas[0,0]
   + 1.315966968302297 betas[0,1] - 1.222949256516093 betas[0,2]
   + 0.7839101778757477 betas[0,3] - betas[0,4] + tau[23,0] = 0
 neg_tau_constr_23_0: tau[23,0] + neg_tau_var_23_0 = 0
 tau_computation_23_1: 1.72852479386002 betas[1,0]
   + 1.315966968302297 betas[1,1] - 1.222949256516093 betas[1,2]
   + 0.7839101778757477 betas[1,3] - betas[1,4] + tau[23,1] = 0
 neg_tau_constr_23_1: tau[23,1] + neg_tau_var_23_1 = 0
 tau_computation_23_2: 1.72852479386002 betas[2,0]
   + 1.315966968302297 betas[2,1] - 1.222949256516093 betas[2,2]
   + 0.7839101778757477 betas[2,3] - betas[2,4] + tau[23,2] = 0
 neg_tau_constr_23_2: tau[23,2] + neg_tau_var_23_2 = 0
 tau_computation_23_3: 1.72852479386002 betas[3,0]
   + 1.315966968302297 betas[3,1] - 1.222949256516093 betas[3,2]
   + 0.7839101778757477 betas[3,3] - betas[3,4] + tau[23,3] = 0
 neg_tau_constr_23_3: tau[23,3] + neg_tau_var_23_3 = 0
 tau_computation_23_4: 1.72852479386002 betas[4,0]
   + 1.315966968302297 betas[4,1] - 1.222949256516093 betas[4,2]
   + 0.7839101778757477 betas[4,3] - betas[4,4] + tau[23,4] = 0
 neg_tau_constr_23_4: tau[23,4] + neg_tau_var_23_4 = 0
 tau_computation_23_5: 1.72852479386002 betas[5,0]
   + 1.315966968302297 betas[5,1] - 1.222949256516093 betas[5,2]
   + 0.7839101778757477 betas[5,3] - betas[5,4] + tau[23,5] = 0
 neg_tau_constr_23_5: tau[23,5] + neg_tau_var_23_5 = 0
 tau_computation_23_6: 1.72852479386002 betas[6,0]
   + 1.315966968302297 betas[6,1] - 1.222949256516093 betas[6,2]
   + 0.7839101778757477 betas[6,3] - betas[6,4] + tau[23,6] = 0
 neg_tau_constr_23_6: tau[23,6] + neg_tau_var_23_6 = 0
 tau_computation_23_7: 1.72852479386002 betas[7,0]
   + 1.315966968302297 betas[7,1] - 1.222949256516093 betas[7,2]
   + 0.7839101778757477 betas[7,3] - betas[7,4] + tau[23,7] = 0
 neg_tau_constr_23_7: tau[23,7] + neg_tau_var_23_7 = 0
 sum_r_23: r[23,0] + r[23,1] + r[23,2] + r[23,3] + r[23,4] + r[23,5]
   + r[23,6] + r[23,7] = 1
 Ensure_only_one_R_true_23: R[23,0] + R[23,1] + R[23,2] + R[23,3] + R[23,4]
   + R[23,5] + R[23,6] + R[23,7] = 1
 Define_Q_hat_23: - 1417.706713512896 R[23,0] - 1386.891008789564 R[23,1]
   - 1384.843226513591 R[23,2] - 1335.012329055854 R[23,3]
   - 1301.494600598587 R[23,4] - 1289.679524751877 R[23,5]
   - 1296.525427455906 R[23,6] - 1303.544560568157 R[23,7] + Q_hat[23] = 0
 Adjust_Q_hat_23: Q0_var[23] - Q_hat[23] + Q_hat_adjusted[23] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_23: Total_Demand_Up_to_K_minus_1[23]
   - 143.8696315710909 R[23,0] - 278.7433143597843 R[23,1]
   - 416.7988172398757 R[23,2] - 551.6468119250783 R[23,3]
   - 667.4707592958373 R[23,4] - 782.2855994481434 R[23,5]
   - 904.5677297796161 R[23,6] - 1033.510955045644 R[23,7] = 0
 Constr_Total_Demand_from_K_to_T_23: Total_Demand_from_k_to_T[23]
   - 1159.520035738079 R[23,0] - 1024.646352949386 R[23,1]
   - 886.5908500692944 R[23,2] - 751.7428553840916 R[23,3]
   - 635.9189080133326 R[23,4] - 521.1040678610266 R[23,5]
   - 398.8219375295539 R[23,6] - 269.8787122635263 R[23,7] = 0
 Constr_Left_0_diff_aux_23: Sold_0[23] - Q0_var[23] + Left_0_aux_23 = 0
 Constr_Lost_0_diff_aux_23: - Total_Demand_Up_to_K_minus_1[23] + Q0_var[23]
   + Lost_0_aux_23 = 0
 Constr_Q1_plus_left_23: - Left_0[23] + Q1_plus_left[23] - Q1_var[23] = 0
 Constr_Left_1_diff_aux_23: Sold_1[23] - Q1_plus_left[23] + Left_0_aux_23
   = 0
 Constr_Lost_1_diff_aux_23: - Total_Demand_from_k_to_T[23]
   + Q1_plus_left[23] + Lost_0_aux_23 = 0
 Profit_Constraint_23: - 500 Sold_0[23] + 500 Lost_0[23] - 500 Sold_1[23]
   + 500 Left_1[23] + 500 Lost_1[23] + profits_vars[23] = 0
 R720: - 0.3411887253336691 alphas[0] + 0.3765967400035515 alphas[1]
   - 0.7477806307898048 alphas[2] - 0.9222149268457147 alphas[3]
   - alphas[4] + f_var[24] = 0
 Q0_upper_bound_24:
   - 1829.241320246946 Fraction_for_second_order_amount[24] + Q0_var[24]
   = 0
 tau_computation_24_0: - 0.3411887253336691 betas[0,0]
   + 0.3765967400035515 betas[0,1] - 0.7477806307898048 betas[0,2]
   - 0.9222149268457147 betas[0,3] - betas[0,4] + tau[24,0] = 0
 neg_tau_constr_24_0: tau[24,0] + neg_tau_var_24_0 = 0
 tau_computation_24_1: - 0.3411887253336691 betas[1,0]
   + 0.3765967400035515 betas[1,1] - 0.7477806307898048 betas[1,2]
   - 0.9222149268457147 betas[1,3] - betas[1,4] + tau[24,1] = 0
 neg_tau_constr_24_1: tau[24,1] + neg_tau_var_24_1 = 0
 tau_computation_24_2: - 0.3411887253336691 betas[2,0]
   + 0.3765967400035515 betas[2,1] - 0.7477806307898048 betas[2,2]
   - 0.9222149268457147 betas[2,3] - betas[2,4] + tau[24,2] = 0
 neg_tau_constr_24_2: tau[24,2] + neg_tau_var_24_2 = 0
 tau_computation_24_3: - 0.3411887253336691 betas[3,0]
   + 0.3765967400035515 betas[3,1] - 0.7477806307898048 betas[3,2]
   - 0.9222149268457147 betas[3,3] - betas[3,4] + tau[24,3] = 0
 neg_tau_constr_24_3: tau[24,3] + neg_tau_var_24_3 = 0
 tau_computation_24_4: - 0.3411887253336691 betas[4,0]
   + 0.3765967400035515 betas[4,1] - 0.7477806307898048 betas[4,2]
   - 0.9222149268457147 betas[4,3] - betas[4,4] + tau[24,4] = 0
 neg_tau_constr_24_4: tau[24,4] + neg_tau_var_24_4 = 0
 tau_computation_24_5: - 0.3411887253336691 betas[5,0]
   + 0.3765967400035515 betas[5,1] - 0.7477806307898048 betas[5,2]
   - 0.9222149268457147 betas[5,3] - betas[5,4] + tau[24,5] = 0
 neg_tau_constr_24_5: tau[24,5] + neg_tau_var_24_5 = 0
 tau_computation_24_6: - 0.3411887253336691 betas[6,0]
   + 0.3765967400035515 betas[6,1] - 0.7477806307898048 betas[6,2]
   - 0.9222149268457147 betas[6,3] - betas[6,4] + tau[24,6] = 0
 neg_tau_constr_24_6: tau[24,6] + neg_tau_var_24_6 = 0
 tau_computation_24_7: - 0.3411887253336691 betas[7,0]
   + 0.3765967400035515 betas[7,1] - 0.7477806307898048 betas[7,2]
   - 0.9222149268457147 betas[7,3] - betas[7,4] + tau[24,7] = 0
 neg_tau_constr_24_7: tau[24,7] + neg_tau_var_24_7 = 0
 sum_r_24: r[24,0] + r[24,1] + r[24,2] + r[24,3] + r[24,4] + r[24,5]
   + r[24,6] + r[24,7] = 1
 Ensure_only_one_R_true_24: R[24,0] + R[24,1] + R[24,2] + R[24,3] + R[24,4]
   + R[24,5] + R[24,6] + R[24,7] = 1
 Define_Q_hat_24: - 1932.018041137182 R[24,0] - 1919.092017039311 R[24,1]
   - 1919.689180893673 R[24,2] - 1936.954945511817 R[24,3]
   - 1936.810518350181 R[24,4] - 1924.389227135358 R[24,5]
   - 1921.914572905625 R[24,6] - 1925.788093451509 R[24,7] + Q_hat[24] = 0
 Adjust_Q_hat_24: Q0_var[24] - Q_hat[24] + Q_hat_adjusted[24] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_24: Total_Demand_Up_to_K_minus_1[24]
   - 194.2426637622111 R[24,0] - 386.3436637430194 R[24,1]
   - 578.0124299128182 R[24,2] - 772.9293862359285 R[24,3]
   - 964.8687841456406 R[24,4] - 1151.020215730493 R[24,5]
   - 1341.495972949021 R[24,6] - 1530.866129762609 R[24,7] = 0
 Constr_Total_Demand_from_K_to_T_24: Total_Demand_from_k_to_T[24]
   - 1738.269136866821 R[24,0] - 1546.168136886012 R[24,1]
   - 1354.499370716213 R[24,2] - 1159.582414393103 R[24,3]
   - 967.6430164833912 R[24,4] - 781.491584898539 R[24,5]
   - 591.0158276800113 R[24,6] - 401.6456708664231 R[24,7] = 0
 Constr_Left_0_diff_aux_24: Sold_0[24] - Q0_var[24] + Left_0_aux_24 = 0
 Constr_Lost_0_diff_aux_24: - Total_Demand_Up_to_K_minus_1[24] + Q0_var[24]
   + Lost_0_aux_24 = 0
 Constr_Q1_plus_left_24: - Left_0[24] + Q1_plus_left[24] - Q1_var[24] = 0
 Constr_Left_1_diff_aux_24: Sold_1[24] - Q1_plus_left[24] + Left_0_aux_24
   = 0
 Constr_Lost_1_diff_aux_24: - Total_Demand_from_k_to_T[24]
   + Q1_plus_left[24] + Lost_0_aux_24 = 0
 Profit_Constraint_24: - 500 Sold_0[24] + 500 Lost_0[24] - 500 Sold_1[24]
   + 500 Left_1[24] + 500 Lost_1[24] + profits_vars[24] = 0
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
 softmax_10_7: - exp_tau_var_10_7 + [ r[10,7] * exp_tau_var_10_0
   + r[10,7] * exp_tau_var_10_1 + r[10,7] * exp_tau_var_10_2
   + r[10,7] * exp_tau_var_10_3 + r[10,7] * exp_tau_var_10_4
   + r[10,7] * exp_tau_var_10_5 + r[10,7] * exp_tau_var_10_6
   + r[10,7] * exp_tau_var_10_7 ] = 0
 softmax_11_7: - exp_tau_var_11_7 + [ r[11,7] * exp_tau_var_11_0
   + r[11,7] * exp_tau_var_11_1 + r[11,7] * exp_tau_var_11_2
   + r[11,7] * exp_tau_var_11_3 + r[11,7] * exp_tau_var_11_4
   + r[11,7] * exp_tau_var_11_5 + r[11,7] * exp_tau_var_11_6
   + r[11,7] * exp_tau_var_11_7 ] = 0
 softmax_12_7: - exp_tau_var_12_7 + [ r[12,7] * exp_tau_var_12_0
   + r[12,7] * exp_tau_var_12_1 + r[12,7] * exp_tau_var_12_2
   + r[12,7] * exp_tau_var_12_3 + r[12,7] * exp_tau_var_12_4
   + r[12,7] * exp_tau_var_12_5 + r[12,7] * exp_tau_var_12_6
   + r[12,7] * exp_tau_var_12_7 ] = 0
 softmax_13_7: - exp_tau_var_13_7 + [ r[13,7] * exp_tau_var_13_0
   + r[13,7] * exp_tau_var_13_1 + r[13,7] * exp_tau_var_13_2
   + r[13,7] * exp_tau_var_13_3 + r[13,7] * exp_tau_var_13_4
   + r[13,7] * exp_tau_var_13_5 + r[13,7] * exp_tau_var_13_6
   + r[13,7] * exp_tau_var_13_7 ] = 0
 softmax_14_7: - exp_tau_var_14_7 + [ r[14,7] * exp_tau_var_14_0
   + r[14,7] * exp_tau_var_14_1 + r[14,7] * exp_tau_var_14_2
   + r[14,7] * exp_tau_var_14_3 + r[14,7] * exp_tau_var_14_4
   + r[14,7] * exp_tau_var_14_5 + r[14,7] * exp_tau_var_14_6
   + r[14,7] * exp_tau_var_14_7 ] = 0
 softmax_15_7: - exp_tau_var_15_7 + [ r[15,7] * exp_tau_var_15_0
   + r[15,7] * exp_tau_var_15_1 + r[15,7] * exp_tau_var_15_2
   + r[15,7] * exp_tau_var_15_3 + r[15,7] * exp_tau_var_15_4
   + r[15,7] * exp_tau_var_15_5 + r[15,7] * exp_tau_var_15_6
   + r[15,7] * exp_tau_var_15_7 ] = 0
 softmax_16_7: - exp_tau_var_16_7 + [ r[16,7] * exp_tau_var_16_0
   + r[16,7] * exp_tau_var_16_1 + r[16,7] * exp_tau_var_16_2
   + r[16,7] * exp_tau_var_16_3 + r[16,7] * exp_tau_var_16_4
   + r[16,7] * exp_tau_var_16_5 + r[16,7] * exp_tau_var_16_6
   + r[16,7] * exp_tau_var_16_7 ] = 0
 softmax_17_7: - exp_tau_var_17_7 + [ r[17,7] * exp_tau_var_17_0
   + r[17,7] * exp_tau_var_17_1 + r[17,7] * exp_tau_var_17_2
   + r[17,7] * exp_tau_var_17_3 + r[17,7] * exp_tau_var_17_4
   + r[17,7] * exp_tau_var_17_5 + r[17,7] * exp_tau_var_17_6
   + r[17,7] * exp_tau_var_17_7 ] = 0
 softmax_18_7: - exp_tau_var_18_7 + [ r[18,7] * exp_tau_var_18_0
   + r[18,7] * exp_tau_var_18_1 + r[18,7] * exp_tau_var_18_2
   + r[18,7] * exp_tau_var_18_3 + r[18,7] * exp_tau_var_18_4
   + r[18,7] * exp_tau_var_18_5 + r[18,7] * exp_tau_var_18_6
   + r[18,7] * exp_tau_var_18_7 ] = 0
 softmax_19_7: - exp_tau_var_19_7 + [ r[19,7] * exp_tau_var_19_0
   + r[19,7] * exp_tau_var_19_1 + r[19,7] * exp_tau_var_19_2
   + r[19,7] * exp_tau_var_19_3 + r[19,7] * exp_tau_var_19_4
   + r[19,7] * exp_tau_var_19_5 + r[19,7] * exp_tau_var_19_6
   + r[19,7] * exp_tau_var_19_7 ] = 0
 softmax_20_7: - exp_tau_var_20_7 + [ r[20,7] * exp_tau_var_20_0
   + r[20,7] * exp_tau_var_20_1 + r[20,7] * exp_tau_var_20_2
   + r[20,7] * exp_tau_var_20_3 + r[20,7] * exp_tau_var_20_4
   + r[20,7] * exp_tau_var_20_5 + r[20,7] * exp_tau_var_20_6
   + r[20,7] * exp_tau_var_20_7 ] = 0
 softmax_21_7: - exp_tau_var_21_7 + [ r[21,7] * exp_tau_var_21_0
   + r[21,7] * exp_tau_var_21_1 + r[21,7] * exp_tau_var_21_2
   + r[21,7] * exp_tau_var_21_3 + r[21,7] * exp_tau_var_21_4
   + r[21,7] * exp_tau_var_21_5 + r[21,7] * exp_tau_var_21_6
   + r[21,7] * exp_tau_var_21_7 ] = 0
 softmax_22_7: - exp_tau_var_22_7 + [ r[22,7] * exp_tau_var_22_0
   + r[22,7] * exp_tau_var_22_1 + r[22,7] * exp_tau_var_22_2
   + r[22,7] * exp_tau_var_22_3 + r[22,7] * exp_tau_var_22_4
   + r[22,7] * exp_tau_var_22_5 + r[22,7] * exp_tau_var_22_6
   + r[22,7] * exp_tau_var_22_7 ] = 0
 softmax_23_7: - exp_tau_var_23_7 + [ r[23,7] * exp_tau_var_23_0
   + r[23,7] * exp_tau_var_23_1 + r[23,7] * exp_tau_var_23_2
   + r[23,7] * exp_tau_var_23_3 + r[23,7] * exp_tau_var_23_4
   + r[23,7] * exp_tau_var_23_5 + r[23,7] * exp_tau_var_23_6
   + r[23,7] * exp_tau_var_23_7 ] = 0
 softmax_24_7: - exp_tau_var_24_7 + [ r[24,7] * exp_tau_var_24_0
   + r[24,7] * exp_tau_var_24_1 + r[24,7] * exp_tau_var_24_2
   + r[24,7] * exp_tau_var_24_3 + r[24,7] * exp_tau_var_24_4
   + r[24,7] * exp_tau_var_24_5 + r[24,7] * exp_tau_var_24_6
   + r[24,7] * exp_tau_var_24_7 ] = 0
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
 GC260: R[10,0] = 1 -> r[10,0] - max_r_helper = 0
 GC261: R[10,1] = 1 -> r[10,1] - max_r_helper = 0
 GC262: R[10,2] = 1 -> r[10,2] - max_r_helper = 0
 GC263: R[10,3] = 1 -> r[10,3] - max_r_helper = 0
 GC264: R[10,4] = 1 -> r[10,4] - max_r_helper = 0
 GC265: R[10,5] = 1 -> r[10,5] - max_r_helper = 0
 GC266: R[10,6] = 1 -> r[10,6] - max_r_helper = 0
 GC267: R[10,7] = 1 -> r[10,7] - max_r_helper = 0
 GC285: R[11,0] = 1 -> r[11,0] - max_r_helper = 0
 GC286: R[11,1] = 1 -> r[11,1] - max_r_helper = 0
 GC287: R[11,2] = 1 -> r[11,2] - max_r_helper = 0
 GC288: R[11,3] = 1 -> r[11,3] - max_r_helper = 0
 GC289: R[11,4] = 1 -> r[11,4] - max_r_helper = 0
 GC290: R[11,5] = 1 -> r[11,5] - max_r_helper = 0
 GC291: R[11,6] = 1 -> r[11,6] - max_r_helper = 0
 GC292: R[11,7] = 1 -> r[11,7] - max_r_helper = 0
 GC310: R[12,0] = 1 -> r[12,0] - max_r_helper = 0
 GC311: R[12,1] = 1 -> r[12,1] - max_r_helper = 0
 GC312: R[12,2] = 1 -> r[12,2] - max_r_helper = 0
 GC313: R[12,3] = 1 -> r[12,3] - max_r_helper = 0
 GC314: R[12,4] = 1 -> r[12,4] - max_r_helper = 0
 GC315: R[12,5] = 1 -> r[12,5] - max_r_helper = 0
 GC316: R[12,6] = 1 -> r[12,6] - max_r_helper = 0
 GC317: R[12,7] = 1 -> r[12,7] - max_r_helper = 0
 GC335: R[13,0] = 1 -> r[13,0] - max_r_helper = 0
 GC336: R[13,1] = 1 -> r[13,1] - max_r_helper = 0
 GC337: R[13,2] = 1 -> r[13,2] - max_r_helper = 0
 GC338: R[13,3] = 1 -> r[13,3] - max_r_helper = 0
 GC339: R[13,4] = 1 -> r[13,4] - max_r_helper = 0
 GC340: R[13,5] = 1 -> r[13,5] - max_r_helper = 0
 GC341: R[13,6] = 1 -> r[13,6] - max_r_helper = 0
 GC342: R[13,7] = 1 -> r[13,7] - max_r_helper = 0
 GC360: R[14,0] = 1 -> r[14,0] - max_r_helper = 0
 GC361: R[14,1] = 1 -> r[14,1] - max_r_helper = 0
 GC362: R[14,2] = 1 -> r[14,2] - max_r_helper = 0
 GC363: R[14,3] = 1 -> r[14,3] - max_r_helper = 0
 GC364: R[14,4] = 1 -> r[14,4] - max_r_helper = 0
 GC365: R[14,5] = 1 -> r[14,5] - max_r_helper = 0
 GC366: R[14,6] = 1 -> r[14,6] - max_r_helper = 0
 GC367: R[14,7] = 1 -> r[14,7] - max_r_helper = 0
 GC385: R[15,0] = 1 -> r[15,0] - max_r_helper = 0
 GC386: R[15,1] = 1 -> r[15,1] - max_r_helper = 0
 GC387: R[15,2] = 1 -> r[15,2] - max_r_helper = 0
 GC388: R[15,3] = 1 -> r[15,3] - max_r_helper = 0
 GC389: R[15,4] = 1 -> r[15,4] - max_r_helper = 0
 GC390: R[15,5] = 1 -> r[15,5] - max_r_helper = 0
 GC391: R[15,6] = 1 -> r[15,6] - max_r_helper = 0
 GC392: R[15,7] = 1 -> r[15,7] - max_r_helper = 0
 GC410: R[16,0] = 1 -> r[16,0] - max_r_helper = 0
 GC411: R[16,1] = 1 -> r[16,1] - max_r_helper = 0
 GC412: R[16,2] = 1 -> r[16,2] - max_r_helper = 0
 GC413: R[16,3] = 1 -> r[16,3] - max_r_helper = 0
 GC414: R[16,4] = 1 -> r[16,4] - max_r_helper = 0
 GC415: R[16,5] = 1 -> r[16,5] - max_r_helper = 0
 GC416: R[16,6] = 1 -> r[16,6] - max_r_helper = 0
 GC417: R[16,7] = 1 -> r[16,7] - max_r_helper = 0
 GC435: R[17,0] = 1 -> r[17,0] - max_r_helper = 0
 GC436: R[17,1] = 1 -> r[17,1] - max_r_helper = 0
 GC437: R[17,2] = 1 -> r[17,2] - max_r_helper = 0
 GC438: R[17,3] = 1 -> r[17,3] - max_r_helper = 0
 GC439: R[17,4] = 1 -> r[17,4] - max_r_helper = 0
 GC440: R[17,5] = 1 -> r[17,5] - max_r_helper = 0
 GC441: R[17,6] = 1 -> r[17,6] - max_r_helper = 0
 GC442: R[17,7] = 1 -> r[17,7] - max_r_helper = 0
 GC460: R[18,0] = 1 -> r[18,0] - max_r_helper = 0
 GC461: R[18,1] = 1 -> r[18,1] - max_r_helper = 0
 GC462: R[18,2] = 1 -> r[18,2] - max_r_helper = 0
 GC463: R[18,3] = 1 -> r[18,3] - max_r_helper = 0
 GC464: R[18,4] = 1 -> r[18,4] - max_r_helper = 0
 GC465: R[18,5] = 1 -> r[18,5] - max_r_helper = 0
 GC466: R[18,6] = 1 -> r[18,6] - max_r_helper = 0
 GC467: R[18,7] = 1 -> r[18,7] - max_r_helper = 0
 GC485: R[19,0] = 1 -> r[19,0] - max_r_helper = 0
 GC486: R[19,1] = 1 -> r[19,1] - max_r_helper = 0
 GC487: R[19,2] = 1 -> r[19,2] - max_r_helper = 0
 GC488: R[19,3] = 1 -> r[19,3] - max_r_helper = 0
 GC489: R[19,4] = 1 -> r[19,4] - max_r_helper = 0
 GC490: R[19,5] = 1 -> r[19,5] - max_r_helper = 0
 GC491: R[19,6] = 1 -> r[19,6] - max_r_helper = 0
 GC492: R[19,7] = 1 -> r[19,7] - max_r_helper = 0
 GC510: R[20,0] = 1 -> r[20,0] - max_r_helper = 0
 GC511: R[20,1] = 1 -> r[20,1] - max_r_helper = 0
 GC512: R[20,2] = 1 -> r[20,2] - max_r_helper = 0
 GC513: R[20,3] = 1 -> r[20,3] - max_r_helper = 0
 GC514: R[20,4] = 1 -> r[20,4] - max_r_helper = 0
 GC515: R[20,5] = 1 -> r[20,5] - max_r_helper = 0
 GC516: R[20,6] = 1 -> r[20,6] - max_r_helper = 0
 GC517: R[20,7] = 1 -> r[20,7] - max_r_helper = 0
 GC535: R[21,0] = 1 -> r[21,0] - max_r_helper = 0
 GC536: R[21,1] = 1 -> r[21,1] - max_r_helper = 0
 GC537: R[21,2] = 1 -> r[21,2] - max_r_helper = 0
 GC538: R[21,3] = 1 -> r[21,3] - max_r_helper = 0
 GC539: R[21,4] = 1 -> r[21,4] - max_r_helper = 0
 GC540: R[21,5] = 1 -> r[21,5] - max_r_helper = 0
 GC541: R[21,6] = 1 -> r[21,6] - max_r_helper = 0
 GC542: R[21,7] = 1 -> r[21,7] - max_r_helper = 0
 GC560: R[22,0] = 1 -> r[22,0] - max_r_helper = 0
 GC561: R[22,1] = 1 -> r[22,1] - max_r_helper = 0
 GC562: R[22,2] = 1 -> r[22,2] - max_r_helper = 0
 GC563: R[22,3] = 1 -> r[22,3] - max_r_helper = 0
 GC564: R[22,4] = 1 -> r[22,4] - max_r_helper = 0
 GC565: R[22,5] = 1 -> r[22,5] - max_r_helper = 0
 GC566: R[22,6] = 1 -> r[22,6] - max_r_helper = 0
 GC567: R[22,7] = 1 -> r[22,7] - max_r_helper = 0
 GC585: R[23,0] = 1 -> r[23,0] - max_r_helper = 0
 GC586: R[23,1] = 1 -> r[23,1] - max_r_helper = 0
 GC587: R[23,2] = 1 -> r[23,2] - max_r_helper = 0
 GC588: R[23,3] = 1 -> r[23,3] - max_r_helper = 0
 GC589: R[23,4] = 1 -> r[23,4] - max_r_helper = 0
 GC590: R[23,5] = 1 -> r[23,5] - max_r_helper = 0
 GC591: R[23,6] = 1 -> r[23,6] - max_r_helper = 0
 GC592: R[23,7] = 1 -> r[23,7] - max_r_helper = 0
 GC610: R[24,0] = 1 -> r[24,0] - max_r_helper = 0
 GC611: R[24,1] = 1 -> r[24,1] - max_r_helper = 0
 GC612: R[24,2] = 1 -> r[24,2] - max_r_helper = 0
 GC613: R[24,3] = 1 -> r[24,3] - max_r_helper = 0
 GC614: R[24,4] = 1 -> r[24,4] - max_r_helper = 0
 GC615: R[24,5] = 1 -> r[24,5] - max_r_helper = 0
 GC616: R[24,6] = 1 -> r[24,6] - max_r_helper = 0
 GC617: R[24,7] = 1 -> r[24,7] - max_r_helper = 0
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
 betas[0,4] free
 betas[1,0] free
 betas[1,1] free
 betas[1,2] free
 betas[1,3] free
 betas[1,4] free
 betas[2,0] free
 betas[2,1] free
 betas[2,2] free
 betas[2,3] free
 betas[2,4] free
 betas[3,0] free
 betas[3,1] free
 betas[3,2] free
 betas[3,3] free
 betas[3,4] free
 betas[4,0] free
 betas[4,1] free
 betas[4,2] free
 betas[4,3] free
 betas[4,4] free
 betas[5,0] free
 betas[5,1] free
 betas[5,2] free
 betas[5,3] free
 betas[5,4] free
 betas[6,0] free
 betas[6,1] free
 betas[6,2] free
 betas[6,3] free
 betas[6,4] free
 betas[7,0] free
 betas[7,1] free
 betas[7,2] free
 betas[7,3] free
 betas[7,4] free
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
 Fraction_for_second_order_amount[10] <= 1
 Fraction_for_second_order_amount[11] <= 1
 Fraction_for_second_order_amount[12] <= 1
 Fraction_for_second_order_amount[13] <= 1
 Fraction_for_second_order_amount[14] <= 1
 Fraction_for_second_order_amount[15] <= 1
 Fraction_for_second_order_amount[16] <= 1
 Fraction_for_second_order_amount[17] <= 1
 Fraction_for_second_order_amount[18] <= 1
 Fraction_for_second_order_amount[19] <= 1
 Fraction_for_second_order_amount[20] <= 1
 Fraction_for_second_order_amount[21] <= 1
 Fraction_for_second_order_amount[22] <= 1
 Fraction_for_second_order_amount[23] <= 1
 Fraction_for_second_order_amount[24] <= 1
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
 tau[10,0] free
 tau[10,1] free
 tau[10,2] free
 tau[10,3] free
 tau[10,4] free
 tau[10,5] free
 tau[10,6] free
 tau[10,7] free
 tau[11,0] free
 tau[11,1] free
 tau[11,2] free
 tau[11,3] free
 tau[11,4] free
 tau[11,5] free
 tau[11,6] free
 tau[11,7] free
 tau[12,0] free
 tau[12,1] free
 tau[12,2] free
 tau[12,3] free
 tau[12,4] free
 tau[12,5] free
 tau[12,6] free
 tau[12,7] free
 tau[13,0] free
 tau[13,1] free
 tau[13,2] free
 tau[13,3] free
 tau[13,4] free
 tau[13,5] free
 tau[13,6] free
 tau[13,7] free
 tau[14,0] free
 tau[14,1] free
 tau[14,2] free
 tau[14,3] free
 tau[14,4] free
 tau[14,5] free
 tau[14,6] free
 tau[14,7] free
 tau[15,0] free
 tau[15,1] free
 tau[15,2] free
 tau[15,3] free
 tau[15,4] free
 tau[15,5] free
 tau[15,6] free
 tau[15,7] free
 tau[16,0] free
 tau[16,1] free
 tau[16,2] free
 tau[16,3] free
 tau[16,4] free
 tau[16,5] free
 tau[16,6] free
 tau[16,7] free
 tau[17,0] free
 tau[17,1] free
 tau[17,2] free
 tau[17,3] free
 tau[17,4] free
 tau[17,5] free
 tau[17,6] free
 tau[17,7] free
 tau[18,0] free
 tau[18,1] free
 tau[18,2] free
 tau[18,3] free
 tau[18,4] free
 tau[18,5] free
 tau[18,6] free
 tau[18,7] free
 tau[19,0] free
 tau[19,1] free
 tau[19,2] free
 tau[19,3] free
 tau[19,4] free
 tau[19,5] free
 tau[19,6] free
 tau[19,7] free
 tau[20,0] free
 tau[20,1] free
 tau[20,2] free
 tau[20,3] free
 tau[20,4] free
 tau[20,5] free
 tau[20,6] free
 tau[20,7] free
 tau[21,0] free
 tau[21,1] free
 tau[21,2] free
 tau[21,3] free
 tau[21,4] free
 tau[21,5] free
 tau[21,6] free
 tau[21,7] free
 tau[22,0] free
 tau[22,1] free
 tau[22,2] free
 tau[22,3] free
 tau[22,4] free
 tau[22,5] free
 tau[22,6] free
 tau[22,7] free
 tau[23,0] free
 tau[23,1] free
 tau[23,2] free
 tau[23,3] free
 tau[23,4] free
 tau[23,5] free
 tau[23,6] free
 tau[23,7] free
 tau[24,0] free
 tau[24,1] free
 tau[24,2] free
 tau[24,3] free
 tau[24,4] free
 tau[24,5] free
 tau[24,6] free
 tau[24,7] free
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
 r[10,0] <= 1
 r[10,1] <= 1
 r[10,2] <= 1
 r[10,3] <= 1
 r[10,4] <= 1
 r[10,5] <= 1
 r[10,6] <= 1
 r[10,7] <= 1
 r[11,0] <= 1
 r[11,1] <= 1
 r[11,2] <= 1
 r[11,3] <= 1
 r[11,4] <= 1
 r[11,5] <= 1
 r[11,6] <= 1
 r[11,7] <= 1
 r[12,0] <= 1
 r[12,1] <= 1
 r[12,2] <= 1
 r[12,3] <= 1
 r[12,4] <= 1
 r[12,5] <= 1
 r[12,6] <= 1
 r[12,7] <= 1
 r[13,0] <= 1
 r[13,1] <= 1
 r[13,2] <= 1
 r[13,3] <= 1
 r[13,4] <= 1
 r[13,5] <= 1
 r[13,6] <= 1
 r[13,7] <= 1
 r[14,0] <= 1
 r[14,1] <= 1
 r[14,2] <= 1
 r[14,3] <= 1
 r[14,4] <= 1
 r[14,5] <= 1
 r[14,6] <= 1
 r[14,7] <= 1
 r[15,0] <= 1
 r[15,1] <= 1
 r[15,2] <= 1
 r[15,3] <= 1
 r[15,4] <= 1
 r[15,5] <= 1
 r[15,6] <= 1
 r[15,7] <= 1
 r[16,0] <= 1
 r[16,1] <= 1
 r[16,2] <= 1
 r[16,3] <= 1
 r[16,4] <= 1
 r[16,5] <= 1
 r[16,6] <= 1
 r[16,7] <= 1
 r[17,0] <= 1
 r[17,1] <= 1
 r[17,2] <= 1
 r[17,3] <= 1
 r[17,4] <= 1
 r[17,5] <= 1
 r[17,6] <= 1
 r[17,7] <= 1
 r[18,0] <= 1
 r[18,1] <= 1
 r[18,2] <= 1
 r[18,3] <= 1
 r[18,4] <= 1
 r[18,5] <= 1
 r[18,6] <= 1
 r[18,7] <= 1
 r[19,0] <= 1
 r[19,1] <= 1
 r[19,2] <= 1
 r[19,3] <= 1
 r[19,4] <= 1
 r[19,5] <= 1
 r[19,6] <= 1
 r[19,7] <= 1
 r[20,0] <= 1
 r[20,1] <= 1
 r[20,2] <= 1
 r[20,3] <= 1
 r[20,4] <= 1
 r[20,5] <= 1
 r[20,6] <= 1
 r[20,7] <= 1
 r[21,0] <= 1
 r[21,1] <= 1
 r[21,2] <= 1
 r[21,3] <= 1
 r[21,4] <= 1
 r[21,5] <= 1
 r[21,6] <= 1
 r[21,7] <= 1
 r[22,0] <= 1
 r[22,1] <= 1
 r[22,2] <= 1
 r[22,3] <= 1
 r[22,4] <= 1
 r[22,5] <= 1
 r[22,6] <= 1
 r[22,7] <= 1
 r[23,0] <= 1
 r[23,1] <= 1
 r[23,2] <= 1
 r[23,3] <= 1
 r[23,4] <= 1
 r[23,5] <= 1
 r[23,6] <= 1
 r[23,7] <= 1
 r[24,0] <= 1
 r[24,1] <= 1
 r[24,2] <= 1
 r[24,3] <= 1
 r[24,4] <= 1
 r[24,5] <= 1
 r[24,6] <= 1
 r[24,7] <= 1
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
 Q_hat_adjusted[10] free
 Q_hat_adjusted[11] free
 Q_hat_adjusted[12] free
 Q_hat_adjusted[13] free
 Q_hat_adjusted[14] free
 Q_hat_adjusted[15] free
 Q_hat_adjusted[16] free
 Q_hat_adjusted[17] free
 Q_hat_adjusted[18] free
 Q_hat_adjusted[19] free
 Q_hat_adjusted[20] free
 Q_hat_adjusted[21] free
 Q_hat_adjusted[22] free
 Q_hat_adjusted[23] free
 Q_hat_adjusted[24] free
 -infinity <= neg_tau_var_0_0 <= 0
 -infinity <= neg_tau_var_0_1 <= 0
 -infinity <= neg_tau_var_0_2 <= 0
 -infinity <= neg_tau_var_0_3 <= 0
 -infinity <= neg_tau_var_0_4 <= 0
 -infinity <= neg_tau_var_0_5 <= 0
 -infinity <= neg_tau_var_0_6 <= 0
 -infinity <= neg_tau_var_0_7 <= 0
 max_r_helper <= 1
 Left_0_aux_0 free
 Lost_0_aux_0 free
 Left_0_aux_0 free
 Lost_0_aux_0 free
 -infinity <= neg_tau_var_1_0 <= 0
 -infinity <= neg_tau_var_1_1 <= 0
 -infinity <= neg_tau_var_1_2 <= 0
 -infinity <= neg_tau_var_1_3 <= 0
 -infinity <= neg_tau_var_1_4 <= 0
 -infinity <= neg_tau_var_1_5 <= 0
 -infinity <= neg_tau_var_1_6 <= 0
 -infinity <= neg_tau_var_1_7 <= 0
 max_r_helper <= 1
 Left_0_aux_1 free
 Lost_0_aux_1 free
 Left_0_aux_1 free
 Lost_0_aux_1 free
 -infinity <= neg_tau_var_2_0 <= 0
 -infinity <= neg_tau_var_2_1 <= 0
 -infinity <= neg_tau_var_2_2 <= 0
 -infinity <= neg_tau_var_2_3 <= 0
 -infinity <= neg_tau_var_2_4 <= 0
 -infinity <= neg_tau_var_2_5 <= 0
 -infinity <= neg_tau_var_2_6 <= 0
 -infinity <= neg_tau_var_2_7 <= 0
 max_r_helper <= 1
 Left_0_aux_2 free
 Lost_0_aux_2 free
 Left_0_aux_2 free
 Lost_0_aux_2 free
 -infinity <= neg_tau_var_3_0 <= 0
 -infinity <= neg_tau_var_3_1 <= 0
 -infinity <= neg_tau_var_3_2 <= 0
 -infinity <= neg_tau_var_3_3 <= 0
 -infinity <= neg_tau_var_3_4 <= 0
 -infinity <= neg_tau_var_3_5 <= 0
 -infinity <= neg_tau_var_3_6 <= 0
 -infinity <= neg_tau_var_3_7 <= 0
 max_r_helper <= 1
 Left_0_aux_3 free
 Lost_0_aux_3 free
 Left_0_aux_3 free
 Lost_0_aux_3 free
 -infinity <= neg_tau_var_4_0 <= 0
 -infinity <= neg_tau_var_4_1 <= 0
 -infinity <= neg_tau_var_4_2 <= 0
 -infinity <= neg_tau_var_4_3 <= 0
 -infinity <= neg_tau_var_4_4 <= 0
 -infinity <= neg_tau_var_4_5 <= 0
 -infinity <= neg_tau_var_4_6 <= 0
 -infinity <= neg_tau_var_4_7 <= 0
 max_r_helper <= 1
 Left_0_aux_4 free
 Lost_0_aux_4 free
 Left_0_aux_4 free
 Lost_0_aux_4 free
 -infinity <= neg_tau_var_5_0 <= 0
 -infinity <= neg_tau_var_5_1 <= 0
 -infinity <= neg_tau_var_5_2 <= 0
 -infinity <= neg_tau_var_5_3 <= 0
 -infinity <= neg_tau_var_5_4 <= 0
 -infinity <= neg_tau_var_5_5 <= 0
 -infinity <= neg_tau_var_5_6 <= 0
 -infinity <= neg_tau_var_5_7 <= 0
 max_r_helper <= 1
 Left_0_aux_5 free
 Lost_0_aux_5 free
 Left_0_aux_5 free
 Lost_0_aux_5 free
 -infinity <= neg_tau_var_6_0 <= 0
 -infinity <= neg_tau_var_6_1 <= 0
 -infinity <= neg_tau_var_6_2 <= 0
 -infinity <= neg_tau_var_6_3 <= 0
 -infinity <= neg_tau_var_6_4 <= 0
 -infinity <= neg_tau_var_6_5 <= 0
 -infinity <= neg_tau_var_6_6 <= 0
 -infinity <= neg_tau_var_6_7 <= 0
 max_r_helper <= 1
 Left_0_aux_6 free
 Lost_0_aux_6 free
 Left_0_aux_6 free
 Lost_0_aux_6 free
 -infinity <= neg_tau_var_7_0 <= 0
 -infinity <= neg_tau_var_7_1 <= 0
 -infinity <= neg_tau_var_7_2 <= 0
 -infinity <= neg_tau_var_7_3 <= 0
 -infinity <= neg_tau_var_7_4 <= 0
 -infinity <= neg_tau_var_7_5 <= 0
 -infinity <= neg_tau_var_7_6 <= 0
 -infinity <= neg_tau_var_7_7 <= 0
 max_r_helper <= 1
 Left_0_aux_7 free
 Lost_0_aux_7 free
 Left_0_aux_7 free
 Lost_0_aux_7 free
 -infinity <= neg_tau_var_8_0 <= 0
 -infinity <= neg_tau_var_8_1 <= 0
 -infinity <= neg_tau_var_8_2 <= 0
 -infinity <= neg_tau_var_8_3 <= 0
 -infinity <= neg_tau_var_8_4 <= 0
 -infinity <= neg_tau_var_8_5 <= 0
 -infinity <= neg_tau_var_8_6 <= 0
 -infinity <= neg_tau_var_8_7 <= 0
 max_r_helper <= 1
 Left_0_aux_8 free
 Lost_0_aux_8 free
 Left_0_aux_8 free
 Lost_0_aux_8 free
 -infinity <= neg_tau_var_9_0 <= 0
 -infinity <= neg_tau_var_9_1 <= 0
 -infinity <= neg_tau_var_9_2 <= 0
 -infinity <= neg_tau_var_9_3 <= 0
 -infinity <= neg_tau_var_9_4 <= 0
 -infinity <= neg_tau_var_9_5 <= 0
 -infinity <= neg_tau_var_9_6 <= 0
 -infinity <= neg_tau_var_9_7 <= 0
 max_r_helper <= 1
 Left_0_aux_9 free
 Lost_0_aux_9 free
 Left_0_aux_9 free
 Lost_0_aux_9 free
 -infinity <= neg_tau_var_10_0 <= 0
 -infinity <= neg_tau_var_10_1 <= 0
 -infinity <= neg_tau_var_10_2 <= 0
 -infinity <= neg_tau_var_10_3 <= 0
 -infinity <= neg_tau_var_10_4 <= 0
 -infinity <= neg_tau_var_10_5 <= 0
 -infinity <= neg_tau_var_10_6 <= 0
 -infinity <= neg_tau_var_10_7 <= 0
 max_r_helper <= 1
 Left_0_aux_10 free
 Lost_0_aux_10 free
 Left_0_aux_10 free
 Lost_0_aux_10 free
 -infinity <= neg_tau_var_11_0 <= 0
 -infinity <= neg_tau_var_11_1 <= 0
 -infinity <= neg_tau_var_11_2 <= 0
 -infinity <= neg_tau_var_11_3 <= 0
 -infinity <= neg_tau_var_11_4 <= 0
 -infinity <= neg_tau_var_11_5 <= 0
 -infinity <= neg_tau_var_11_6 <= 0
 -infinity <= neg_tau_var_11_7 <= 0
 max_r_helper <= 1
 Left_0_aux_11 free
 Lost_0_aux_11 free
 Left_0_aux_11 free
 Lost_0_aux_11 free
 -infinity <= neg_tau_var_12_0 <= 0
 -infinity <= neg_tau_var_12_1 <= 0
 -infinity <= neg_tau_var_12_2 <= 0
 -infinity <= neg_tau_var_12_3 <= 0
 -infinity <= neg_tau_var_12_4 <= 0
 -infinity <= neg_tau_var_12_5 <= 0
 -infinity <= neg_tau_var_12_6 <= 0
 -infinity <= neg_tau_var_12_7 <= 0
 max_r_helper <= 1
 Left_0_aux_12 free
 Lost_0_aux_12 free
 Left_0_aux_12 free
 Lost_0_aux_12 free
 -infinity <= neg_tau_var_13_0 <= 0
 -infinity <= neg_tau_var_13_1 <= 0
 -infinity <= neg_tau_var_13_2 <= 0
 -infinity <= neg_tau_var_13_3 <= 0
 -infinity <= neg_tau_var_13_4 <= 0
 -infinity <= neg_tau_var_13_5 <= 0
 -infinity <= neg_tau_var_13_6 <= 0
 -infinity <= neg_tau_var_13_7 <= 0
 max_r_helper <= 1
 Left_0_aux_13 free
 Lost_0_aux_13 free
 Left_0_aux_13 free
 Lost_0_aux_13 free
 -infinity <= neg_tau_var_14_0 <= 0
 -infinity <= neg_tau_var_14_1 <= 0
 -infinity <= neg_tau_var_14_2 <= 0
 -infinity <= neg_tau_var_14_3 <= 0
 -infinity <= neg_tau_var_14_4 <= 0
 -infinity <= neg_tau_var_14_5 <= 0
 -infinity <= neg_tau_var_14_6 <= 0
 -infinity <= neg_tau_var_14_7 <= 0
 max_r_helper <= 1
 Left_0_aux_14 free
 Lost_0_aux_14 free
 Left_0_aux_14 free
 Lost_0_aux_14 free
 -infinity <= neg_tau_var_15_0 <= 0
 -infinity <= neg_tau_var_15_1 <= 0
 -infinity <= neg_tau_var_15_2 <= 0
 -infinity <= neg_tau_var_15_3 <= 0
 -infinity <= neg_tau_var_15_4 <= 0
 -infinity <= neg_tau_var_15_5 <= 0
 -infinity <= neg_tau_var_15_6 <= 0
 -infinity <= neg_tau_var_15_7 <= 0
 max_r_helper <= 1
 Left_0_aux_15 free
 Lost_0_aux_15 free
 Left_0_aux_15 free
 Lost_0_aux_15 free
 -infinity <= neg_tau_var_16_0 <= 0
 -infinity <= neg_tau_var_16_1 <= 0
 -infinity <= neg_tau_var_16_2 <= 0
 -infinity <= neg_tau_var_16_3 <= 0
 -infinity <= neg_tau_var_16_4 <= 0
 -infinity <= neg_tau_var_16_5 <= 0
 -infinity <= neg_tau_var_16_6 <= 0
 -infinity <= neg_tau_var_16_7 <= 0
 max_r_helper <= 1
 Left_0_aux_16 free
 Lost_0_aux_16 free
 Left_0_aux_16 free
 Lost_0_aux_16 free
 -infinity <= neg_tau_var_17_0 <= 0
 -infinity <= neg_tau_var_17_1 <= 0
 -infinity <= neg_tau_var_17_2 <= 0
 -infinity <= neg_tau_var_17_3 <= 0
 -infinity <= neg_tau_var_17_4 <= 0
 -infinity <= neg_tau_var_17_5 <= 0
 -infinity <= neg_tau_var_17_6 <= 0
 -infinity <= neg_tau_var_17_7 <= 0
 max_r_helper <= 1
 Left_0_aux_17 free
 Lost_0_aux_17 free
 Left_0_aux_17 free
 Lost_0_aux_17 free
 -infinity <= neg_tau_var_18_0 <= 0
 -infinity <= neg_tau_var_18_1 <= 0
 -infinity <= neg_tau_var_18_2 <= 0
 -infinity <= neg_tau_var_18_3 <= 0
 -infinity <= neg_tau_var_18_4 <= 0
 -infinity <= neg_tau_var_18_5 <= 0
 -infinity <= neg_tau_var_18_6 <= 0
 -infinity <= neg_tau_var_18_7 <= 0
 max_r_helper <= 1
 Left_0_aux_18 free
 Lost_0_aux_18 free
 Left_0_aux_18 free
 Lost_0_aux_18 free
 -infinity <= neg_tau_var_19_0 <= 0
 -infinity <= neg_tau_var_19_1 <= 0
 -infinity <= neg_tau_var_19_2 <= 0
 -infinity <= neg_tau_var_19_3 <= 0
 -infinity <= neg_tau_var_19_4 <= 0
 -infinity <= neg_tau_var_19_5 <= 0
 -infinity <= neg_tau_var_19_6 <= 0
 -infinity <= neg_tau_var_19_7 <= 0
 max_r_helper <= 1
 Left_0_aux_19 free
 Lost_0_aux_19 free
 Left_0_aux_19 free
 Lost_0_aux_19 free
 -infinity <= neg_tau_var_20_0 <= 0
 -infinity <= neg_tau_var_20_1 <= 0
 -infinity <= neg_tau_var_20_2 <= 0
 -infinity <= neg_tau_var_20_3 <= 0
 -infinity <= neg_tau_var_20_4 <= 0
 -infinity <= neg_tau_var_20_5 <= 0
 -infinity <= neg_tau_var_20_6 <= 0
 -infinity <= neg_tau_var_20_7 <= 0
 max_r_helper <= 1
 Left_0_aux_20 free
 Lost_0_aux_20 free
 Left_0_aux_20 free
 Lost_0_aux_20 free
 -infinity <= neg_tau_var_21_0 <= 0
 -infinity <= neg_tau_var_21_1 <= 0
 -infinity <= neg_tau_var_21_2 <= 0
 -infinity <= neg_tau_var_21_3 <= 0
 -infinity <= neg_tau_var_21_4 <= 0
 -infinity <= neg_tau_var_21_5 <= 0
 -infinity <= neg_tau_var_21_6 <= 0
 -infinity <= neg_tau_var_21_7 <= 0
 max_r_helper <= 1
 Left_0_aux_21 free
 Lost_0_aux_21 free
 Left_0_aux_21 free
 Lost_0_aux_21 free
 -infinity <= neg_tau_var_22_0 <= 0
 -infinity <= neg_tau_var_22_1 <= 0
 -infinity <= neg_tau_var_22_2 <= 0
 -infinity <= neg_tau_var_22_3 <= 0
 -infinity <= neg_tau_var_22_4 <= 0
 -infinity <= neg_tau_var_22_5 <= 0
 -infinity <= neg_tau_var_22_6 <= 0
 -infinity <= neg_tau_var_22_7 <= 0
 max_r_helper <= 1
 Left_0_aux_22 free
 Lost_0_aux_22 free
 Left_0_aux_22 free
 Lost_0_aux_22 free
 -infinity <= neg_tau_var_23_0 <= 0
 -infinity <= neg_tau_var_23_1 <= 0
 -infinity <= neg_tau_var_23_2 <= 0
 -infinity <= neg_tau_var_23_3 <= 0
 -infinity <= neg_tau_var_23_4 <= 0
 -infinity <= neg_tau_var_23_5 <= 0
 -infinity <= neg_tau_var_23_6 <= 0
 -infinity <= neg_tau_var_23_7 <= 0
 max_r_helper <= 1
 Left_0_aux_23 free
 Lost_0_aux_23 free
 Left_0_aux_23 free
 Lost_0_aux_23 free
 -infinity <= neg_tau_var_24_0 <= 0
 -infinity <= neg_tau_var_24_1 <= 0
 -infinity <= neg_tau_var_24_2 <= 0
 -infinity <= neg_tau_var_24_3 <= 0
 -infinity <= neg_tau_var_24_4 <= 0
 -infinity <= neg_tau_var_24_5 <= 0
 -infinity <= neg_tau_var_24_6 <= 0
 -infinity <= neg_tau_var_24_7 <= 0
 max_r_helper <= 1
 Left_0_aux_24 free
 Lost_0_aux_24 free
 Left_0_aux_24 free
 Lost_0_aux_24 free
Binaries
 R[0,0] R[0,1] R[0,2] R[0,3] R[0,4] R[0,5] R[0,6] R[0,7] R[1,0] R[1,1]
 R[1,2] R[1,3] R[1,4] R[1,5] R[1,6] R[1,7] R[2,0] R[2,1] R[2,2] R[2,3]
 R[2,4] R[2,5] R[2,6] R[2,7] R[3,0] R[3,1] R[3,2] R[3,3] R[3,4] R[3,5]
 R[3,6] R[3,7] R[4,0] R[4,1] R[4,2] R[4,3] R[4,4] R[4,5] R[4,6] R[4,7]
 R[5,0] R[5,1] R[5,2] R[5,3] R[5,4] R[5,5] R[5,6] R[5,7] R[6,0] R[6,1]
 R[6,2] R[6,3] R[6,4] R[6,5] R[6,6] R[6,7] R[7,0] R[7,1] R[7,2] R[7,3]
 R[7,4] R[7,5] R[7,6] R[7,7] R[8,0] R[8,1] R[8,2] R[8,3] R[8,4] R[8,5]
 R[8,6] R[8,7] R[9,0] R[9,1] R[9,2] R[9,3] R[9,4] R[9,5] R[9,6] R[9,7]
 R[10,0] R[10,1] R[10,2] R[10,3] R[10,4] R[10,5] R[10,6] R[10,7] R[11,0]
 R[11,1] R[11,2] R[11,3] R[11,4] R[11,5] R[11,6] R[11,7] R[12,0] R[12,1]
 R[12,2] R[12,3] R[12,4] R[12,5] R[12,6] R[12,7] R[13,0] R[13,1] R[13,2]
 R[13,3] R[13,4] R[13,5] R[13,6] R[13,7] R[14,0] R[14,1] R[14,2] R[14,3]
 R[14,4] R[14,5] R[14,6] R[14,7] R[15,0] R[15,1] R[15,2] R[15,3] R[15,4]
 R[15,5] R[15,6] R[15,7] R[16,0] R[16,1] R[16,2] R[16,3] R[16,4] R[16,5]
 R[16,6] R[16,7] R[17,0] R[17,1] R[17,2] R[17,3] R[17,4] R[17,5] R[17,6]
 R[17,7] R[18,0] R[18,1] R[18,2] R[18,3] R[18,4] R[18,5] R[18,6] R[18,7]
 R[19,0] R[19,1] R[19,2] R[19,3] R[19,4] R[19,5] R[19,6] R[19,7] R[20,0]
 R[20,1] R[20,2] R[20,3] R[20,4] R[20,5] R[20,6] R[20,7] R[21,0] R[21,1]
 R[21,2] R[21,3] R[21,4] R[21,5] R[21,6] R[21,7] R[22,0] R[22,1] R[22,2]
 R[22,3] R[22,4] R[22,5] R[22,6] R[22,7] R[23,0] R[23,1] R[23,2] R[23,3]
 R[23,4] R[23,5] R[23,6] R[23,7] R[24,0] R[24,1] R[24,2] R[24,3] R[24,4]
 R[24,5] R[24,6] R[24,7]
General Constraints
 logistic_constraint_0: Fraction_for_second_order_amount[0] =
   LOGISTIC ( f_var[0] )
 GC1: exp_tau_var_0_0 = EXP ( neg_tau_var_0_0 )
 GC2: exp_tau_var_0_1 = EXP ( neg_tau_var_0_1 )
 GC3: exp_tau_var_0_2 = EXP ( neg_tau_var_0_2 )
 GC4: exp_tau_var_0_3 = EXP ( neg_tau_var_0_3 )
 GC5: exp_tau_var_0_4 = EXP ( neg_tau_var_0_4 )
 GC6: exp_tau_var_0_5 = EXP ( neg_tau_var_0_5 )
 GC7: exp_tau_var_0_6 = EXP ( neg_tau_var_0_6 )
 GC8: exp_tau_var_0_7 = EXP ( neg_tau_var_0_7 )
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
 GC26: exp_tau_var_1_0 = EXP ( neg_tau_var_1_0 )
 GC27: exp_tau_var_1_1 = EXP ( neg_tau_var_1_1 )
 GC28: exp_tau_var_1_2 = EXP ( neg_tau_var_1_2 )
 GC29: exp_tau_var_1_3 = EXP ( neg_tau_var_1_3 )
 GC30: exp_tau_var_1_4 = EXP ( neg_tau_var_1_4 )
 GC31: exp_tau_var_1_5 = EXP ( neg_tau_var_1_5 )
 GC32: exp_tau_var_1_6 = EXP ( neg_tau_var_1_6 )
 GC33: exp_tau_var_1_7 = EXP ( neg_tau_var_1_7 )
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
 GC51: exp_tau_var_2_0 = EXP ( neg_tau_var_2_0 )
 GC52: exp_tau_var_2_1 = EXP ( neg_tau_var_2_1 )
 GC53: exp_tau_var_2_2 = EXP ( neg_tau_var_2_2 )
 GC54: exp_tau_var_2_3 = EXP ( neg_tau_var_2_3 )
 GC55: exp_tau_var_2_4 = EXP ( neg_tau_var_2_4 )
 GC56: exp_tau_var_2_5 = EXP ( neg_tau_var_2_5 )
 GC57: exp_tau_var_2_6 = EXP ( neg_tau_var_2_6 )
 GC58: exp_tau_var_2_7 = EXP ( neg_tau_var_2_7 )
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
 GC76: exp_tau_var_3_0 = EXP ( neg_tau_var_3_0 )
 GC77: exp_tau_var_3_1 = EXP ( neg_tau_var_3_1 )
 GC78: exp_tau_var_3_2 = EXP ( neg_tau_var_3_2 )
 GC79: exp_tau_var_3_3 = EXP ( neg_tau_var_3_3 )
 GC80: exp_tau_var_3_4 = EXP ( neg_tau_var_3_4 )
 GC81: exp_tau_var_3_5 = EXP ( neg_tau_var_3_5 )
 GC82: exp_tau_var_3_6 = EXP ( neg_tau_var_3_6 )
 GC83: exp_tau_var_3_7 = EXP ( neg_tau_var_3_7 )
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
 GC101: exp_tau_var_4_0 = EXP ( neg_tau_var_4_0 )
 GC102: exp_tau_var_4_1 = EXP ( neg_tau_var_4_1 )
 GC103: exp_tau_var_4_2 = EXP ( neg_tau_var_4_2 )
 GC104: exp_tau_var_4_3 = EXP ( neg_tau_var_4_3 )
 GC105: exp_tau_var_4_4 = EXP ( neg_tau_var_4_4 )
 GC106: exp_tau_var_4_5 = EXP ( neg_tau_var_4_5 )
 GC107: exp_tau_var_4_6 = EXP ( neg_tau_var_4_6 )
 GC108: exp_tau_var_4_7 = EXP ( neg_tau_var_4_7 )
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
 GC126: exp_tau_var_5_0 = EXP ( neg_tau_var_5_0 )
 GC127: exp_tau_var_5_1 = EXP ( neg_tau_var_5_1 )
 GC128: exp_tau_var_5_2 = EXP ( neg_tau_var_5_2 )
 GC129: exp_tau_var_5_3 = EXP ( neg_tau_var_5_3 )
 GC130: exp_tau_var_5_4 = EXP ( neg_tau_var_5_4 )
 GC131: exp_tau_var_5_5 = EXP ( neg_tau_var_5_5 )
 GC132: exp_tau_var_5_6 = EXP ( neg_tau_var_5_6 )
 GC133: exp_tau_var_5_7 = EXP ( neg_tau_var_5_7 )
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
 GC151: exp_tau_var_6_0 = EXP ( neg_tau_var_6_0 )
 GC152: exp_tau_var_6_1 = EXP ( neg_tau_var_6_1 )
 GC153: exp_tau_var_6_2 = EXP ( neg_tau_var_6_2 )
 GC154: exp_tau_var_6_3 = EXP ( neg_tau_var_6_3 )
 GC155: exp_tau_var_6_4 = EXP ( neg_tau_var_6_4 )
 GC156: exp_tau_var_6_5 = EXP ( neg_tau_var_6_5 )
 GC157: exp_tau_var_6_6 = EXP ( neg_tau_var_6_6 )
 GC158: exp_tau_var_6_7 = EXP ( neg_tau_var_6_7 )
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
 GC176: exp_tau_var_7_0 = EXP ( neg_tau_var_7_0 )
 GC177: exp_tau_var_7_1 = EXP ( neg_tau_var_7_1 )
 GC178: exp_tau_var_7_2 = EXP ( neg_tau_var_7_2 )
 GC179: exp_tau_var_7_3 = EXP ( neg_tau_var_7_3 )
 GC180: exp_tau_var_7_4 = EXP ( neg_tau_var_7_4 )
 GC181: exp_tau_var_7_5 = EXP ( neg_tau_var_7_5 )
 GC182: exp_tau_var_7_6 = EXP ( neg_tau_var_7_6 )
 GC183: exp_tau_var_7_7 = EXP ( neg_tau_var_7_7 )
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
 GC201: exp_tau_var_8_0 = EXP ( neg_tau_var_8_0 )
 GC202: exp_tau_var_8_1 = EXP ( neg_tau_var_8_1 )
 GC203: exp_tau_var_8_2 = EXP ( neg_tau_var_8_2 )
 GC204: exp_tau_var_8_3 = EXP ( neg_tau_var_8_3 )
 GC205: exp_tau_var_8_4 = EXP ( neg_tau_var_8_4 )
 GC206: exp_tau_var_8_5 = EXP ( neg_tau_var_8_5 )
 GC207: exp_tau_var_8_6 = EXP ( neg_tau_var_8_6 )
 GC208: exp_tau_var_8_7 = EXP ( neg_tau_var_8_7 )
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
 GC226: exp_tau_var_9_0 = EXP ( neg_tau_var_9_0 )
 GC227: exp_tau_var_9_1 = EXP ( neg_tau_var_9_1 )
 GC228: exp_tau_var_9_2 = EXP ( neg_tau_var_9_2 )
 GC229: exp_tau_var_9_3 = EXP ( neg_tau_var_9_3 )
 GC230: exp_tau_var_9_4 = EXP ( neg_tau_var_9_4 )
 GC231: exp_tau_var_9_5 = EXP ( neg_tau_var_9_5 )
 GC232: exp_tau_var_9_6 = EXP ( neg_tau_var_9_6 )
 GC233: exp_tau_var_9_7 = EXP ( neg_tau_var_9_7 )
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
 logistic_constraint_10: Fraction_for_second_order_amount[10] =
   LOGISTIC ( f_var[10] )
 GC251: exp_tau_var_10_0 = EXP ( neg_tau_var_10_0 )
 GC252: exp_tau_var_10_1 = EXP ( neg_tau_var_10_1 )
 GC253: exp_tau_var_10_2 = EXP ( neg_tau_var_10_2 )
 GC254: exp_tau_var_10_3 = EXP ( neg_tau_var_10_3 )
 GC255: exp_tau_var_10_4 = EXP ( neg_tau_var_10_4 )
 GC256: exp_tau_var_10_5 = EXP ( neg_tau_var_10_5 )
 GC257: exp_tau_var_10_6 = EXP ( neg_tau_var_10_6 )
 GC258: exp_tau_var_10_7 = EXP ( neg_tau_var_10_7 )
 MaxRConstraint_10: max_r_helper = MAX ( r[10,0] , r[10,1] , r[10,2] ,
   r[10,3] , r[10,4] , r[10,5] , r[10,6] , r[10,7] )
 Max_Constraint_10: Q1_var[10] = MAX ( Q_hat_adjusted[10] , 0 )
 Constr_Sold_0_min_10: Sold_0[10] = MIN (
   Total_Demand_Up_to_K_minus_1[10] , Q0_var[10] )
 Constr_Left_0_max_10: Left_0[10] = MAX ( Left_0_aux_10 , 0 )
 Constr_Lost_0_max_10: Lost_0[10] = MAX ( Lost_0_aux_10 , 0 )
 Constr_Sold_1_min_10: Sold_1[10] = MIN ( Total_Demand_from_k_to_T[10] ,
   Q1_plus_left[10] )
 Constr_Left_1_max_10: Left_1[10] = MAX ( Left_0_aux_10 , 0 )
 Constr_Lost_1_max_10: Lost_1[10] = MAX ( Lost_0_aux_10 , 0 )
 logistic_constraint_11: Fraction_for_second_order_amount[11] =
   LOGISTIC ( f_var[11] )
 GC276: exp_tau_var_11_0 = EXP ( neg_tau_var_11_0 )
 GC277: exp_tau_var_11_1 = EXP ( neg_tau_var_11_1 )
 GC278: exp_tau_var_11_2 = EXP ( neg_tau_var_11_2 )
 GC279: exp_tau_var_11_3 = EXP ( neg_tau_var_11_3 )
 GC280: exp_tau_var_11_4 = EXP ( neg_tau_var_11_4 )
 GC281: exp_tau_var_11_5 = EXP ( neg_tau_var_11_5 )
 GC282: exp_tau_var_11_6 = EXP ( neg_tau_var_11_6 )
 GC283: exp_tau_var_11_7 = EXP ( neg_tau_var_11_7 )
 MaxRConstraint_11: max_r_helper = MAX ( r[11,0] , r[11,1] , r[11,2] ,
   r[11,3] , r[11,4] , r[11,5] , r[11,6] , r[11,7] )
 Max_Constraint_11: Q1_var[11] = MAX ( Q_hat_adjusted[11] , 0 )
 Constr_Sold_0_min_11: Sold_0[11] = MIN (
   Total_Demand_Up_to_K_minus_1[11] , Q0_var[11] )
 Constr_Left_0_max_11: Left_0[11] = MAX ( Left_0_aux_11 , 0 )
 Constr_Lost_0_max_11: Lost_0[11] = MAX ( Lost_0_aux_11 , 0 )
 Constr_Sold_1_min_11: Sold_1[11] = MIN ( Total_Demand_from_k_to_T[11] ,
   Q1_plus_left[11] )
 Constr_Left_1_max_11: Left_1[11] = MAX ( Left_0_aux_11 , 0 )
 Constr_Lost_1_max_11: Lost_1[11] = MAX ( Lost_0_aux_11 , 0 )
 logistic_constraint_12: Fraction_for_second_order_amount[12] =
   LOGISTIC ( f_var[12] )
 GC301: exp_tau_var_12_0 = EXP ( neg_tau_var_12_0 )
 GC302: exp_tau_var_12_1 = EXP ( neg_tau_var_12_1 )
 GC303: exp_tau_var_12_2 = EXP ( neg_tau_var_12_2 )
 GC304: exp_tau_var_12_3 = EXP ( neg_tau_var_12_3 )
 GC305: exp_tau_var_12_4 = EXP ( neg_tau_var_12_4 )
 GC306: exp_tau_var_12_5 = EXP ( neg_tau_var_12_5 )
 GC307: exp_tau_var_12_6 = EXP ( neg_tau_var_12_6 )
 GC308: exp_tau_var_12_7 = EXP ( neg_tau_var_12_7 )
 MaxRConstraint_12: max_r_helper = MAX ( r[12,0] , r[12,1] , r[12,2] ,
   r[12,3] , r[12,4] , r[12,5] , r[12,6] , r[12,7] )
 Max_Constraint_12: Q1_var[12] = MAX ( Q_hat_adjusted[12] , 0 )
 Constr_Sold_0_min_12: Sold_0[12] = MIN (
   Total_Demand_Up_to_K_minus_1[12] , Q0_var[12] )
 Constr_Left_0_max_12: Left_0[12] = MAX ( Left_0_aux_12 , 0 )
 Constr_Lost_0_max_12: Lost_0[12] = MAX ( Lost_0_aux_12 , 0 )
 Constr_Sold_1_min_12: Sold_1[12] = MIN ( Total_Demand_from_k_to_T[12] ,
   Q1_plus_left[12] )
 Constr_Left_1_max_12: Left_1[12] = MAX ( Left_0_aux_12 , 0 )
 Constr_Lost_1_max_12: Lost_1[12] = MAX ( Lost_0_aux_12 , 0 )
 logistic_constraint_13: Fraction_for_second_order_amount[13] =
   LOGISTIC ( f_var[13] )
 GC326: exp_tau_var_13_0 = EXP ( neg_tau_var_13_0 )
 GC327: exp_tau_var_13_1 = EXP ( neg_tau_var_13_1 )
 GC328: exp_tau_var_13_2 = EXP ( neg_tau_var_13_2 )
 GC329: exp_tau_var_13_3 = EXP ( neg_tau_var_13_3 )
 GC330: exp_tau_var_13_4 = EXP ( neg_tau_var_13_4 )
 GC331: exp_tau_var_13_5 = EXP ( neg_tau_var_13_5 )
 GC332: exp_tau_var_13_6 = EXP ( neg_tau_var_13_6 )
 GC333: exp_tau_var_13_7 = EXP ( neg_tau_var_13_7 )
 MaxRConstraint_13: max_r_helper = MAX ( r[13,0] , r[13,1] , r[13,2] ,
   r[13,3] , r[13,4] , r[13,5] , r[13,6] , r[13,7] )
 Max_Constraint_13: Q1_var[13] = MAX ( Q_hat_adjusted[13] , 0 )
 Constr_Sold_0_min_13: Sold_0[13] = MIN (
   Total_Demand_Up_to_K_minus_1[13] , Q0_var[13] )
 Constr_Left_0_max_13: Left_0[13] = MAX ( Left_0_aux_13 , 0 )
 Constr_Lost_0_max_13: Lost_0[13] = MAX ( Lost_0_aux_13 , 0 )
 Constr_Sold_1_min_13: Sold_1[13] = MIN ( Total_Demand_from_k_to_T[13] ,
   Q1_plus_left[13] )
 Constr_Left_1_max_13: Left_1[13] = MAX ( Left_0_aux_13 , 0 )
 Constr_Lost_1_max_13: Lost_1[13] = MAX ( Lost_0_aux_13 , 0 )
 logistic_constraint_14: Fraction_for_second_order_amount[14] =
   LOGISTIC ( f_var[14] )
 GC351: exp_tau_var_14_0 = EXP ( neg_tau_var_14_0 )
 GC352: exp_tau_var_14_1 = EXP ( neg_tau_var_14_1 )
 GC353: exp_tau_var_14_2 = EXP ( neg_tau_var_14_2 )
 GC354: exp_tau_var_14_3 = EXP ( neg_tau_var_14_3 )
 GC355: exp_tau_var_14_4 = EXP ( neg_tau_var_14_4 )
 GC356: exp_tau_var_14_5 = EXP ( neg_tau_var_14_5 )
 GC357: exp_tau_var_14_6 = EXP ( neg_tau_var_14_6 )
 GC358: exp_tau_var_14_7 = EXP ( neg_tau_var_14_7 )
 MaxRConstraint_14: max_r_helper = MAX ( r[14,0] , r[14,1] , r[14,2] ,
   r[14,3] , r[14,4] , r[14,5] , r[14,6] , r[14,7] )
 Max_Constraint_14: Q1_var[14] = MAX ( Q_hat_adjusted[14] , 0 )
 Constr_Sold_0_min_14: Sold_0[14] = MIN (
   Total_Demand_Up_to_K_minus_1[14] , Q0_var[14] )
 Constr_Left_0_max_14: Left_0[14] = MAX ( Left_0_aux_14 , 0 )
 Constr_Lost_0_max_14: Lost_0[14] = MAX ( Lost_0_aux_14 , 0 )
 Constr_Sold_1_min_14: Sold_1[14] = MIN ( Total_Demand_from_k_to_T[14] ,
   Q1_plus_left[14] )
 Constr_Left_1_max_14: Left_1[14] = MAX ( Left_0_aux_14 , 0 )
 Constr_Lost_1_max_14: Lost_1[14] = MAX ( Lost_0_aux_14 , 0 )
 logistic_constraint_15: Fraction_for_second_order_amount[15] =
   LOGISTIC ( f_var[15] )
 GC376: exp_tau_var_15_0 = EXP ( neg_tau_var_15_0 )
 GC377: exp_tau_var_15_1 = EXP ( neg_tau_var_15_1 )
 GC378: exp_tau_var_15_2 = EXP ( neg_tau_var_15_2 )
 GC379: exp_tau_var_15_3 = EXP ( neg_tau_var_15_3 )
 GC380: exp_tau_var_15_4 = EXP ( neg_tau_var_15_4 )
 GC381: exp_tau_var_15_5 = EXP ( neg_tau_var_15_5 )
 GC382: exp_tau_var_15_6 = EXP ( neg_tau_var_15_6 )
 GC383: exp_tau_var_15_7 = EXP ( neg_tau_var_15_7 )
 MaxRConstraint_15: max_r_helper = MAX ( r[15,0] , r[15,1] , r[15,2] ,
   r[15,3] , r[15,4] , r[15,5] , r[15,6] , r[15,7] )
 Max_Constraint_15: Q1_var[15] = MAX ( Q_hat_adjusted[15] , 0 )
 Constr_Sold_0_min_15: Sold_0[15] = MIN (
   Total_Demand_Up_to_K_minus_1[15] , Q0_var[15] )
 Constr_Left_0_max_15: Left_0[15] = MAX ( Left_0_aux_15 , 0 )
 Constr_Lost_0_max_15: Lost_0[15] = MAX ( Lost_0_aux_15 , 0 )
 Constr_Sold_1_min_15: Sold_1[15] = MIN ( Total_Demand_from_k_to_T[15] ,
   Q1_plus_left[15] )
 Constr_Left_1_max_15: Left_1[15] = MAX ( Left_0_aux_15 , 0 )
 Constr_Lost_1_max_15: Lost_1[15] = MAX ( Lost_0_aux_15 , 0 )
 logistic_constraint_16: Fraction_for_second_order_amount[16] =
   LOGISTIC ( f_var[16] )
 GC401: exp_tau_var_16_0 = EXP ( neg_tau_var_16_0 )
 GC402: exp_tau_var_16_1 = EXP ( neg_tau_var_16_1 )
 GC403: exp_tau_var_16_2 = EXP ( neg_tau_var_16_2 )
 GC404: exp_tau_var_16_3 = EXP ( neg_tau_var_16_3 )
 GC405: exp_tau_var_16_4 = EXP ( neg_tau_var_16_4 )
 GC406: exp_tau_var_16_5 = EXP ( neg_tau_var_16_5 )
 GC407: exp_tau_var_16_6 = EXP ( neg_tau_var_16_6 )
 GC408: exp_tau_var_16_7 = EXP ( neg_tau_var_16_7 )
 MaxRConstraint_16: max_r_helper = MAX ( r[16,0] , r[16,1] , r[16,2] ,
   r[16,3] , r[16,4] , r[16,5] , r[16,6] , r[16,7] )
 Max_Constraint_16: Q1_var[16] = MAX ( Q_hat_adjusted[16] , 0 )
 Constr_Sold_0_min_16: Sold_0[16] = MIN (
   Total_Demand_Up_to_K_minus_1[16] , Q0_var[16] )
 Constr_Left_0_max_16: Left_0[16] = MAX ( Left_0_aux_16 , 0 )
 Constr_Lost_0_max_16: Lost_0[16] = MAX ( Lost_0_aux_16 , 0 )
 Constr_Sold_1_min_16: Sold_1[16] = MIN ( Total_Demand_from_k_to_T[16] ,
   Q1_plus_left[16] )
 Constr_Left_1_max_16: Left_1[16] = MAX ( Left_0_aux_16 , 0 )
 Constr_Lost_1_max_16: Lost_1[16] = MAX ( Lost_0_aux_16 , 0 )
 logistic_constraint_17: Fraction_for_second_order_amount[17] =
   LOGISTIC ( f_var[17] )
 GC426: exp_tau_var_17_0 = EXP ( neg_tau_var_17_0 )
 GC427: exp_tau_var_17_1 = EXP ( neg_tau_var_17_1 )
 GC428: exp_tau_var_17_2 = EXP ( neg_tau_var_17_2 )
 GC429: exp_tau_var_17_3 = EXP ( neg_tau_var_17_3 )
 GC430: exp_tau_var_17_4 = EXP ( neg_tau_var_17_4 )
 GC431: exp_tau_var_17_5 = EXP ( neg_tau_var_17_5 )
 GC432: exp_tau_var_17_6 = EXP ( neg_tau_var_17_6 )
 GC433: exp_tau_var_17_7 = EXP ( neg_tau_var_17_7 )
 MaxRConstraint_17: max_r_helper = MAX ( r[17,0] , r[17,1] , r[17,2] ,
   r[17,3] , r[17,4] , r[17,5] , r[17,6] , r[17,7] )
 Max_Constraint_17: Q1_var[17] = MAX ( Q_hat_adjusted[17] , 0 )
 Constr_Sold_0_min_17: Sold_0[17] = MIN (
   Total_Demand_Up_to_K_minus_1[17] , Q0_var[17] )
 Constr_Left_0_max_17: Left_0[17] = MAX ( Left_0_aux_17 , 0 )
 Constr_Lost_0_max_17: Lost_0[17] = MAX ( Lost_0_aux_17 , 0 )
 Constr_Sold_1_min_17: Sold_1[17] = MIN ( Total_Demand_from_k_to_T[17] ,
   Q1_plus_left[17] )
 Constr_Left_1_max_17: Left_1[17] = MAX ( Left_0_aux_17 , 0 )
 Constr_Lost_1_max_17: Lost_1[17] = MAX ( Lost_0_aux_17 , 0 )
 logistic_constraint_18: Fraction_for_second_order_amount[18] =
   LOGISTIC ( f_var[18] )
 GC451: exp_tau_var_18_0 = EXP ( neg_tau_var_18_0 )
 GC452: exp_tau_var_18_1 = EXP ( neg_tau_var_18_1 )
 GC453: exp_tau_var_18_2 = EXP ( neg_tau_var_18_2 )
 GC454: exp_tau_var_18_3 = EXP ( neg_tau_var_18_3 )
 GC455: exp_tau_var_18_4 = EXP ( neg_tau_var_18_4 )
 GC456: exp_tau_var_18_5 = EXP ( neg_tau_var_18_5 )
 GC457: exp_tau_var_18_6 = EXP ( neg_tau_var_18_6 )
 GC458: exp_tau_var_18_7 = EXP ( neg_tau_var_18_7 )
 MaxRConstraint_18: max_r_helper = MAX ( r[18,0] , r[18,1] , r[18,2] ,
   r[18,3] , r[18,4] , r[18,5] , r[18,6] , r[18,7] )
 Max_Constraint_18: Q1_var[18] = MAX ( Q_hat_adjusted[18] , 0 )
 Constr_Sold_0_min_18: Sold_0[18] = MIN (
   Total_Demand_Up_to_K_minus_1[18] , Q0_var[18] )
 Constr_Left_0_max_18: Left_0[18] = MAX ( Left_0_aux_18 , 0 )
 Constr_Lost_0_max_18: Lost_0[18] = MAX ( Lost_0_aux_18 , 0 )
 Constr_Sold_1_min_18: Sold_1[18] = MIN ( Total_Demand_from_k_to_T[18] ,
   Q1_plus_left[18] )
 Constr_Left_1_max_18: Left_1[18] = MAX ( Left_0_aux_18 , 0 )
 Constr_Lost_1_max_18: Lost_1[18] = MAX ( Lost_0_aux_18 , 0 )
 logistic_constraint_19: Fraction_for_second_order_amount[19] =
   LOGISTIC ( f_var[19] )
 GC476: exp_tau_var_19_0 = EXP ( neg_tau_var_19_0 )
 GC477: exp_tau_var_19_1 = EXP ( neg_tau_var_19_1 )
 GC478: exp_tau_var_19_2 = EXP ( neg_tau_var_19_2 )
 GC479: exp_tau_var_19_3 = EXP ( neg_tau_var_19_3 )
 GC480: exp_tau_var_19_4 = EXP ( neg_tau_var_19_4 )
 GC481: exp_tau_var_19_5 = EXP ( neg_tau_var_19_5 )
 GC482: exp_tau_var_19_6 = EXP ( neg_tau_var_19_6 )
 GC483: exp_tau_var_19_7 = EXP ( neg_tau_var_19_7 )
 MaxRConstraint_19: max_r_helper = MAX ( r[19,0] , r[19,1] , r[19,2] ,
   r[19,3] , r[19,4] , r[19,5] , r[19,6] , r[19,7] )
 Max_Constraint_19: Q1_var[19] = MAX ( Q_hat_adjusted[19] , 0 )
 Constr_Sold_0_min_19: Sold_0[19] = MIN (
   Total_Demand_Up_to_K_minus_1[19] , Q0_var[19] )
 Constr_Left_0_max_19: Left_0[19] = MAX ( Left_0_aux_19 , 0 )
 Constr_Lost_0_max_19: Lost_0[19] = MAX ( Lost_0_aux_19 , 0 )
 Constr_Sold_1_min_19: Sold_1[19] = MIN ( Total_Demand_from_k_to_T[19] ,
   Q1_plus_left[19] )
 Constr_Left_1_max_19: Left_1[19] = MAX ( Left_0_aux_19 , 0 )
 Constr_Lost_1_max_19: Lost_1[19] = MAX ( Lost_0_aux_19 , 0 )
 logistic_constraint_20: Fraction_for_second_order_amount[20] =
   LOGISTIC ( f_var[20] )
 GC501: exp_tau_var_20_0 = EXP ( neg_tau_var_20_0 )
 GC502: exp_tau_var_20_1 = EXP ( neg_tau_var_20_1 )
 GC503: exp_tau_var_20_2 = EXP ( neg_tau_var_20_2 )
 GC504: exp_tau_var_20_3 = EXP ( neg_tau_var_20_3 )
 GC505: exp_tau_var_20_4 = EXP ( neg_tau_var_20_4 )
 GC506: exp_tau_var_20_5 = EXP ( neg_tau_var_20_5 )
 GC507: exp_tau_var_20_6 = EXP ( neg_tau_var_20_6 )
 GC508: exp_tau_var_20_7 = EXP ( neg_tau_var_20_7 )
 MaxRConstraint_20: max_r_helper = MAX ( r[20,0] , r[20,1] , r[20,2] ,
   r[20,3] , r[20,4] , r[20,5] , r[20,6] , r[20,7] )
 Max_Constraint_20: Q1_var[20] = MAX ( Q_hat_adjusted[20] , 0 )
 Constr_Sold_0_min_20: Sold_0[20] = MIN (
   Total_Demand_Up_to_K_minus_1[20] , Q0_var[20] )
 Constr_Left_0_max_20: Left_0[20] = MAX ( Left_0_aux_20 , 0 )
 Constr_Lost_0_max_20: Lost_0[20] = MAX ( Lost_0_aux_20 , 0 )
 Constr_Sold_1_min_20: Sold_1[20] = MIN ( Total_Demand_from_k_to_T[20] ,
   Q1_plus_left[20] )
 Constr_Left_1_max_20: Left_1[20] = MAX ( Left_0_aux_20 , 0 )
 Constr_Lost_1_max_20: Lost_1[20] = MAX ( Lost_0_aux_20 , 0 )
 logistic_constraint_21: Fraction_for_second_order_amount[21] =
   LOGISTIC ( f_var[21] )
 GC526: exp_tau_var_21_0 = EXP ( neg_tau_var_21_0 )
 GC527: exp_tau_var_21_1 = EXP ( neg_tau_var_21_1 )
 GC528: exp_tau_var_21_2 = EXP ( neg_tau_var_21_2 )
 GC529: exp_tau_var_21_3 = EXP ( neg_tau_var_21_3 )
 GC530: exp_tau_var_21_4 = EXP ( neg_tau_var_21_4 )
 GC531: exp_tau_var_21_5 = EXP ( neg_tau_var_21_5 )
 GC532: exp_tau_var_21_6 = EXP ( neg_tau_var_21_6 )
 GC533: exp_tau_var_21_7 = EXP ( neg_tau_var_21_7 )
 MaxRConstraint_21: max_r_helper = MAX ( r[21,0] , r[21,1] , r[21,2] ,
   r[21,3] , r[21,4] , r[21,5] , r[21,6] , r[21,7] )
 Max_Constraint_21: Q1_var[21] = MAX ( Q_hat_adjusted[21] , 0 )
 Constr_Sold_0_min_21: Sold_0[21] = MIN (
   Total_Demand_Up_to_K_minus_1[21] , Q0_var[21] )
 Constr_Left_0_max_21: Left_0[21] = MAX ( Left_0_aux_21 , 0 )
 Constr_Lost_0_max_21: Lost_0[21] = MAX ( Lost_0_aux_21 , 0 )
 Constr_Sold_1_min_21: Sold_1[21] = MIN ( Total_Demand_from_k_to_T[21] ,
   Q1_plus_left[21] )
 Constr_Left_1_max_21: Left_1[21] = MAX ( Left_0_aux_21 , 0 )
 Constr_Lost_1_max_21: Lost_1[21] = MAX ( Lost_0_aux_21 , 0 )
 logistic_constraint_22: Fraction_for_second_order_amount[22] =
   LOGISTIC ( f_var[22] )
 GC551: exp_tau_var_22_0 = EXP ( neg_tau_var_22_0 )
 GC552: exp_tau_var_22_1 = EXP ( neg_tau_var_22_1 )
 GC553: exp_tau_var_22_2 = EXP ( neg_tau_var_22_2 )
 GC554: exp_tau_var_22_3 = EXP ( neg_tau_var_22_3 )
 GC555: exp_tau_var_22_4 = EXP ( neg_tau_var_22_4 )
 GC556: exp_tau_var_22_5 = EXP ( neg_tau_var_22_5 )
 GC557: exp_tau_var_22_6 = EXP ( neg_tau_var_22_6 )
 GC558: exp_tau_var_22_7 = EXP ( neg_tau_var_22_7 )
 MaxRConstraint_22: max_r_helper = MAX ( r[22,0] , r[22,1] , r[22,2] ,
   r[22,3] , r[22,4] , r[22,5] , r[22,6] , r[22,7] )
 Max_Constraint_22: Q1_var[22] = MAX ( Q_hat_adjusted[22] , 0 )
 Constr_Sold_0_min_22: Sold_0[22] = MIN (
   Total_Demand_Up_to_K_minus_1[22] , Q0_var[22] )
 Constr_Left_0_max_22: Left_0[22] = MAX ( Left_0_aux_22 , 0 )
 Constr_Lost_0_max_22: Lost_0[22] = MAX ( Lost_0_aux_22 , 0 )
 Constr_Sold_1_min_22: Sold_1[22] = MIN ( Total_Demand_from_k_to_T[22] ,
   Q1_plus_left[22] )
 Constr_Left_1_max_22: Left_1[22] = MAX ( Left_0_aux_22 , 0 )
 Constr_Lost_1_max_22: Lost_1[22] = MAX ( Lost_0_aux_22 , 0 )
 logistic_constraint_23: Fraction_for_second_order_amount[23] =
   LOGISTIC ( f_var[23] )
 GC576: exp_tau_var_23_0 = EXP ( neg_tau_var_23_0 )
 GC577: exp_tau_var_23_1 = EXP ( neg_tau_var_23_1 )
 GC578: exp_tau_var_23_2 = EXP ( neg_tau_var_23_2 )
 GC579: exp_tau_var_23_3 = EXP ( neg_tau_var_23_3 )
 GC580: exp_tau_var_23_4 = EXP ( neg_tau_var_23_4 )
 GC581: exp_tau_var_23_5 = EXP ( neg_tau_var_23_5 )
 GC582: exp_tau_var_23_6 = EXP ( neg_tau_var_23_6 )
 GC583: exp_tau_var_23_7 = EXP ( neg_tau_var_23_7 )
 MaxRConstraint_23: max_r_helper = MAX ( r[23,0] , r[23,1] , r[23,2] ,
   r[23,3] , r[23,4] , r[23,5] , r[23,6] , r[23,7] )
 Max_Constraint_23: Q1_var[23] = MAX ( Q_hat_adjusted[23] , 0 )
 Constr_Sold_0_min_23: Sold_0[23] = MIN (
   Total_Demand_Up_to_K_minus_1[23] , Q0_var[23] )
 Constr_Left_0_max_23: Left_0[23] = MAX ( Left_0_aux_23 , 0 )
 Constr_Lost_0_max_23: Lost_0[23] = MAX ( Lost_0_aux_23 , 0 )
 Constr_Sold_1_min_23: Sold_1[23] = MIN ( Total_Demand_from_k_to_T[23] ,
   Q1_plus_left[23] )
 Constr_Left_1_max_23: Left_1[23] = MAX ( Left_0_aux_23 , 0 )
 Constr_Lost_1_max_23: Lost_1[23] = MAX ( Lost_0_aux_23 , 0 )
 logistic_constraint_24: Fraction_for_second_order_amount[24] =
   LOGISTIC ( f_var[24] )
 GC601: exp_tau_var_24_0 = EXP ( neg_tau_var_24_0 )
 GC602: exp_tau_var_24_1 = EXP ( neg_tau_var_24_1 )
 GC603: exp_tau_var_24_2 = EXP ( neg_tau_var_24_2 )
 GC604: exp_tau_var_24_3 = EXP ( neg_tau_var_24_3 )
 GC605: exp_tau_var_24_4 = EXP ( neg_tau_var_24_4 )
 GC606: exp_tau_var_24_5 = EXP ( neg_tau_var_24_5 )
 GC607: exp_tau_var_24_6 = EXP ( neg_tau_var_24_6 )
 GC608: exp_tau_var_24_7 = EXP ( neg_tau_var_24_7 )
 MaxRConstraint_24: max_r_helper = MAX ( r[24,0] , r[24,1] , r[24,2] ,
   r[24,3] , r[24,4] , r[24,5] , r[24,6] , r[24,7] )
 Max_Constraint_24: Q1_var[24] = MAX ( Q_hat_adjusted[24] , 0 )
 Constr_Sold_0_min_24: Sold_0[24] = MIN (
   Total_Demand_Up_to_K_minus_1[24] , Q0_var[24] )
 Constr_Left_0_max_24: Left_0[24] = MAX ( Left_0_aux_24 , 0 )
 Constr_Lost_0_max_24: Lost_0[24] = MAX ( Lost_0_aux_24 , 0 )
 Constr_Sold_1_min_24: Sold_1[24] = MIN ( Total_Demand_from_k_to_T[24] ,
   Q1_plus_left[24] )
 Constr_Left_1_max_24: Left_1[24] = MAX ( Left_0_aux_24 , 0 )
 Constr_Lost_1_max_24: Lost_1[24] = MAX ( Lost_0_aux_24 , 0 )
End
