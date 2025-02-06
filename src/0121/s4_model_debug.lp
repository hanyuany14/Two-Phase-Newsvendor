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
   + profits_vars[23] + profits_vars[24] + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
Subject To
 R0: 0.1485442135232429 alphas[0] - 0.3851892822364052 alphas[1]
   - 0.2394304242396365 alphas[2] + 1.04415860753828 alphas[3] - alphas[4]
   + f_var[0] = 0
 Q0_upper_bound_0: - 2053.400566507637 Fraction_for_second_order_amount[0]
   + Q0_var[0] = 0
 tau_computation_0_0: - betas[0] + 0.1485442135232429 gammas[0]
   - 0.3851892822364052 gammas[1] - 0.2394304242396365 gammas[2]
   + 1.04415860753828 gammas[3] + tau[0,0] = 0
 tau_computation_0_1: - betas[1] + 0.1485442135232429 gammas[0]
   - 0.3851892822364052 gammas[1] - 0.2394304242396365 gammas[2]
   + 1.04415860753828 gammas[3] + tau[0,1] = 0
 tau_computation_0_2: - betas[2] + 0.1485442135232429 gammas[0]
   - 0.3851892822364052 gammas[1] - 0.2394304242396365 gammas[2]
   + 1.04415860753828 gammas[3] + tau[0,2] = 0
 tau_computation_0_3: - betas[3] + 0.1485442135232429 gammas[0]
   - 0.3851892822364052 gammas[1] - 0.2394304242396365 gammas[2]
   + 1.04415860753828 gammas[3] + tau[0,3] = 0
 tau_computation_0_4: - betas[4] + 0.1485442135232429 gammas[0]
   - 0.3851892822364052 gammas[1] - 0.2394304242396365 gammas[2]
   + 1.04415860753828 gammas[3] + tau[0,4] = 0
 tau_computation_0_5: - betas[5] + 0.1485442135232429 gammas[0]
   - 0.3851892822364052 gammas[1] - 0.2394304242396365 gammas[2]
   + 1.04415860753828 gammas[3] + tau[0,5] = 0
 tau_computation_0_6: - betas[6] + 0.1485442135232429 gammas[0]
   - 0.3851892822364052 gammas[1] - 0.2394304242396365 gammas[2]
   + 1.04415860753828 gammas[3] + tau[0,6] = 0
 tau_computation_0_7: - betas[7] + 0.1485442135232429 gammas[0]
   - 0.3851892822364052 gammas[1] - 0.2394304242396365 gammas[2]
   + 1.04415860753828 gammas[3] + tau[0,7] = 0
 Ensure_only_one_R_true_0: R[0,0] + R[0,1] + R[0,2] + R[0,3] + R[0,4]
   + R[0,5] + R[0,6] + R[0,7] = 1
 Define_Q_hat_0: - 1803.591606637637 R[0,0] - 2160.074729141367 R[0,1]
   - 1970.308492506769 R[0,2] - 2006.985752143723 R[0,3]
   - 2161.906582761115 R[0,4] - 2161.906378443357 R[0,5]
   - 2034.129315029104 R[0,6] - 2064.951581619729 R[0,7] + Q_hat[0] = 0
 Adjust_Q_hat_0: Q0_var[0] - Q_hat[0] + Q_hat_adjusted[0] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_0: Total_Demand_Up_to_K_minus_1[0]
   - 134.0391723472468 R[0,0] - 353.4932995524932 R[0,1]
   - 353.4932995524932 R[0,2] - 572.9474333741139 R[0,3]
   - 1031.315057880999 R[0,4] - 1250.769192080056 R[0,5]
   - 1461.270222653054 R[0,6] - 1626.043314427274 R[0,7] = 0
 Constr_Total_Demand_from_K_to_T_0: Total_Demand_from_k_to_T[0]
   - 1930.912410478143 R[0,0] - 1711.458283272897 R[0,1]
   - 1711.458283272896 R[0,2] - 1492.004149451276 R[0,3]
   - 1033.636524944391 R[0,4] - 814.1823907453332 R[0,5]
   - 603.6813601723356 R[0,6] - 438.9082683981151 R[0,7] = 0
 Constr_Left_0_diff_aux_0: Sold_0[0] - Q0_var[0] + Left_0_aux_0 = 0
 Constr_Lost_0_diff_aux_0: - Total_Demand_Up_to_K_minus_1[0] + Q0_var[0]
   + Lost_0_aux_0 = 0
 Constr_Q1_plus_left_0: - Left_0[0] + Q1_plus_left[0] - Q1_var[0] = 0
 Constr_Left_1_diff_aux_0: Sold_1[0] - Q1_plus_left[0] + Left_0_aux_0 = 0
 Constr_Lost_1_diff_aux_0: - Total_Demand_from_k_to_T[0] + Q1_plus_left[0]
   + Lost_0_aux_0 = 0
 Profit_Constraint_0: - 500 Sold_0[0] + 500 Lost_0[0] - 500 Sold_1[0]
   + 500 Left_1[0] + 500 Lost_1[0] + profits_vars[0] = 0
 R21: - 0.4405511694217167 alphas[0] + 0.0486297156342791 alphas[1]
   + 1.57241865352909 alphas[2] - 1.211868098505336 alphas[3] - alphas[4]
   + f_var[1] = 0
 Q0_upper_bound_1: - 2053.400566507637 Fraction_for_second_order_amount[1]
   + Q0_var[1] = 0
 tau_computation_1_0: - betas[0] - 0.4405511694217167 gammas[0]
   + 0.0486297156342791 gammas[1] + 1.57241865352909 gammas[2]
   - 1.211868098505336 gammas[3] + tau[1,0] = 0
 tau_computation_1_1: - betas[1] - 0.4405511694217167 gammas[0]
   + 0.0486297156342791 gammas[1] + 1.57241865352909 gammas[2]
   - 1.211868098505336 gammas[3] + tau[1,1] = 0
 tau_computation_1_2: - betas[2] - 0.4405511694217167 gammas[0]
   + 0.0486297156342791 gammas[1] + 1.57241865352909 gammas[2]
   - 1.211868098505336 gammas[3] + tau[1,2] = 0
 tau_computation_1_3: - betas[3] - 0.4405511694217167 gammas[0]
   + 0.0486297156342791 gammas[1] + 1.57241865352909 gammas[2]
   - 1.211868098505336 gammas[3] + tau[1,3] = 0
 tau_computation_1_4: - betas[4] - 0.4405511694217167 gammas[0]
   + 0.0486297156342791 gammas[1] + 1.57241865352909 gammas[2]
   - 1.211868098505336 gammas[3] + tau[1,4] = 0
 tau_computation_1_5: - betas[5] - 0.4405511694217167 gammas[0]
   + 0.0486297156342791 gammas[1] + 1.57241865352909 gammas[2]
   - 1.211868098505336 gammas[3] + tau[1,5] = 0
 tau_computation_1_6: - betas[6] - 0.4405511694217167 gammas[0]
   + 0.0486297156342791 gammas[1] + 1.57241865352909 gammas[2]
   - 1.211868098505336 gammas[3] + tau[1,6] = 0
 tau_computation_1_7: - betas[7] - 0.4405511694217167 gammas[0]
   + 0.0486297156342791 gammas[1] + 1.57241865352909 gammas[2]
   - 1.211868098505336 gammas[3] + tau[1,7] = 0
 Ensure_only_one_R_true_1: R[1,0] + R[1,1] + R[1,2] + R[1,3] + R[1,4]
   + R[1,5] + R[1,6] + R[1,7] = 1
 Define_Q_hat_1: - 1764.351318994491 R[1,0] - 1204.150006475395 R[1,1]
   - 1090.881693387844 R[1,2] - 1087.4465078343 R[1,3]
   - 1036.783329534061 R[1,4] - 1036.783678177365 R[1,5]
   - 1030.140724939198 R[1,6] - 974.5031660793892 R[1,7] + Q_hat[1] = 0
 Adjust_Q_hat_1: Q0_var[1] - Q_hat[1] + Q_hat_adjusted[1] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_1: Total_Demand_Up_to_K_minus_1[1]
   - 119.0307870885745 R[1,0] - 236.5281439553683 R[1,1]
   - 260.0708963234039 R[1,2] - 377.5728898339252 R[1,3]
   - 469.707656178634 R[1,4] - 587.2096494766752 R[1,5]
   - 739.4991742575022 R[1,6] - 739.4991742575022 R[1,7] = 0
 Constr_Total_Demand_from_K_to_T_1: Total_Demand_from_k_to_T[1]
   - 855.4723737650104 R[1,0] - 737.9750168982165 R[1,1]
   - 714.4322645301809 R[1,2] - 596.9302710196596 R[1,3]
   - 504.7955046749508 R[1,4] - 387.2935113769096 R[1,5]
   - 235.0039865960826 R[1,6] - 235.0039865960826 R[1,7] = 0
 Constr_Left_0_diff_aux_1: Sold_0[1] - Q0_var[1] + Left_0_aux_1 = 0
 Constr_Lost_0_diff_aux_1: - Total_Demand_Up_to_K_minus_1[1] + Q0_var[1]
   + Lost_0_aux_1 = 0
 Constr_Q1_plus_left_1: - Left_0[1] + Q1_plus_left[1] - Q1_var[1] = 0
 Constr_Left_1_diff_aux_1: Sold_1[1] - Q1_plus_left[1] + Left_0_aux_1 = 0
 Constr_Lost_1_diff_aux_1: - Total_Demand_from_k_to_T[1] + Q1_plus_left[1]
   + Lost_0_aux_1 = 0
 Profit_Constraint_1: - 500 Sold_0[1] + 500 Lost_0[1] - 500 Sold_1[1]
   + 500 Left_1[1] + 500 Lost_1[1] + profits_vars[1] = 0
 R42: - 0.0424788118340407 alphas[0] - 1.763758393415958 alphas[1]
   - 0.4943523487684089 alphas[2] + 1.003209748914306 alphas[3] - alphas[4]
   + f_var[2] = 0
 Q0_upper_bound_2: - 2053.400566507637 Fraction_for_second_order_amount[2]
   + Q0_var[2] = 0
 tau_computation_2_0: - betas[0] - 0.0424788118340407 gammas[0]
   - 1.763758393415958 gammas[1] - 0.4943523487684089 gammas[2]
   + 1.003209748914306 gammas[3] + tau[2,0] = 0
 tau_computation_2_1: - betas[1] - 0.0424788118340407 gammas[0]
   - 1.763758393415958 gammas[1] - 0.4943523487684089 gammas[2]
   + 1.003209748914306 gammas[3] + tau[2,1] = 0
 tau_computation_2_2: - betas[2] - 0.0424788118340407 gammas[0]
   - 1.763758393415958 gammas[1] - 0.4943523487684089 gammas[2]
   + 1.003209748914306 gammas[3] + tau[2,2] = 0
 tau_computation_2_3: - betas[3] - 0.0424788118340407 gammas[0]
   - 1.763758393415958 gammas[1] - 0.4943523487684089 gammas[2]
   + 1.003209748914306 gammas[3] + tau[2,3] = 0
 tau_computation_2_4: - betas[4] - 0.0424788118340407 gammas[0]
   - 1.763758393415958 gammas[1] - 0.4943523487684089 gammas[2]
   + 1.003209748914306 gammas[3] + tau[2,4] = 0
 tau_computation_2_5: - betas[5] - 0.0424788118340407 gammas[0]
   - 1.763758393415958 gammas[1] - 0.4943523487684089 gammas[2]
   + 1.003209748914306 gammas[3] + tau[2,5] = 0
 tau_computation_2_6: - betas[6] - 0.0424788118340407 gammas[0]
   - 1.763758393415958 gammas[1] - 0.4943523487684089 gammas[2]
   + 1.003209748914306 gammas[3] + tau[2,6] = 0
 tau_computation_2_7: - betas[7] - 0.0424788118340407 gammas[0]
   - 1.763758393415958 gammas[1] - 0.4943523487684089 gammas[2]
   + 1.003209748914306 gammas[3] + tau[2,7] = 0
 Ensure_only_one_R_true_2: R[2,0] + R[2,1] + R[2,2] + R[2,3] + R[2,4]
   + R[2,5] + R[2,6] + R[2,7] = 1
 Define_Q_hat_2: - 2578.375340720291 R[2,0] - 2353.841604824432 R[2,1]
   - 2299.534880635418 R[2,2] - 2312.804559184591 R[2,3]
   - 2208.290323699796 R[2,4] - 2208.290647685527 R[2,5]
   - 2071.285029355398 R[2,6] - 2066.117285596097 R[2,7] + Q_hat[2] = 0
 Adjust_Q_hat_2: Q0_var[2] - Q_hat[2] + Q_hat_adjusted[2] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_2: Total_Demand_Up_to_K_minus_1[2]
   - 430.3737170634652 R[2,0] - 664.1722340118447 R[2,1]
   - 985.2864749888029 R[2,2] - 1219.084980742378 R[2,3]
   - 1219.084980742378 R[2,4] - 1452.883486273027 R[2,5]
   - 1491.225382191153 R[2,6] - 1598.520274925257 R[2,7] = 0
 Constr_Total_Demand_from_K_to_T_2: Total_Demand_from_k_to_T[2]
   - 1635.743568923089 R[2,0] - 1401.945051974709 R[2,1]
   - 1080.830810997751 R[2,2] - 847.0323052441757 R[2,3]
   - 847.0323052441757 R[2,4] - 613.233799713527 R[2,5]
   - 574.8919037954014 R[2,6] - 467.5970110612975 R[2,7] = 0
 Constr_Left_0_diff_aux_2: Sold_0[2] - Q0_var[2] + Left_0_aux_2 = 0
 Constr_Lost_0_diff_aux_2: - Total_Demand_Up_to_K_minus_1[2] + Q0_var[2]
   + Lost_0_aux_2 = 0
 Constr_Q1_plus_left_2: - Left_0[2] + Q1_plus_left[2] - Q1_var[2] = 0
 Constr_Left_1_diff_aux_2: Sold_1[2] - Q1_plus_left[2] + Left_0_aux_2 = 0
 Constr_Lost_1_diff_aux_2: - Total_Demand_from_k_to_T[2] + Q1_plus_left[2]
   + Lost_0_aux_2 = 0
 Profit_Constraint_2: - 500 Sold_0[2] + 500 Lost_0[2] - 500 Sold_1[2]
   + 500 Left_1[2] + 500 Lost_1[2] + profits_vars[2] = 0
 R63: 0.1624604984913878 alphas[0] + 1.158127845551552 alphas[1]
   - 1.502864317011586 alphas[2] - 0.4190277033472191 alphas[3] - alphas[4]
   + f_var[3] = 0
 Q0_upper_bound_3: - 2053.400566507637 Fraction_for_second_order_amount[3]
   + Q0_var[3] = 0
 tau_computation_3_0: - betas[0] + 0.1624604984913878 gammas[0]
   + 1.158127845551552 gammas[1] - 1.502864317011586 gammas[2]
   - 0.4190277033472191 gammas[3] + tau[3,0] = 0
 tau_computation_3_1: - betas[1] + 0.1624604984913878 gammas[0]
   + 1.158127845551552 gammas[1] - 1.502864317011586 gammas[2]
   - 0.4190277033472191 gammas[3] + tau[3,1] = 0
 tau_computation_3_2: - betas[2] + 0.1624604984913878 gammas[0]
   + 1.158127845551552 gammas[1] - 1.502864317011586 gammas[2]
   - 0.4190277033472191 gammas[3] + tau[3,2] = 0
 tau_computation_3_3: - betas[3] + 0.1624604984913878 gammas[0]
   + 1.158127845551552 gammas[1] - 1.502864317011586 gammas[2]
   - 0.4190277033472191 gammas[3] + tau[3,3] = 0
 tau_computation_3_4: - betas[4] + 0.1624604984913878 gammas[0]
   + 1.158127845551552 gammas[1] - 1.502864317011586 gammas[2]
   - 0.4190277033472191 gammas[3] + tau[3,4] = 0
 tau_computation_3_5: - betas[5] + 0.1624604984913878 gammas[0]
   + 1.158127845551552 gammas[1] - 1.502864317011586 gammas[2]
   - 0.4190277033472191 gammas[3] + tau[3,5] = 0
 tau_computation_3_6: - betas[6] + 0.1624604984913878 gammas[0]
   + 1.158127845551552 gammas[1] - 1.502864317011586 gammas[2]
   - 0.4190277033472191 gammas[3] + tau[3,6] = 0
 tau_computation_3_7: - betas[7] + 0.1624604984913878 gammas[0]
   + 1.158127845551552 gammas[1] - 1.502864317011586 gammas[2]
   - 0.4190277033472191 gammas[3] + tau[3,7] = 0
 Ensure_only_one_R_true_3: R[3,0] + R[3,1] + R[3,2] + R[3,3] + R[3,4]
   + R[3,5] + R[3,6] + R[3,7] = 1
 Define_Q_hat_3: - 2226.65277696469 R[3,0] - 2857.118820641231 R[3,1]
   - 2839.970801349355 R[3,2] - 2829.253372673651 R[3,3]
   - 2745.372260279675 R[3,4] - 2745.371727606611 R[3,5]
   - 2890.991112230085 R[3,6] - 2979.896608353331 R[3,7] + Q_hat[3] = 0
 Adjust_Q_hat_3: Q0_var[3] - Q_hat[3] + Q_hat_adjusted[3] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_3: Total_Demand_Up_to_K_minus_1[3]
   - 295.8490217040011 R[3,0] - 586.3961581028016 R[3,1]
   - 858.5323003292507 R[3,2] - 1149.079436340925 R[3,3]
   - 1336.568307592816 R[3,4] - 1627.115443872176 R[3,5]
   - 2043.463500983727 R[3,6] - 2398.802338453156 R[3,7] = 0
 Constr_Total_Demand_from_K_to_T_3: Total_Demand_from_k_to_T[3]
   - 2684.047589307874 R[3,0] - 2393.500452909073 R[3,1]
   - 2121.364310682624 R[3,2] - 1830.81717467095 R[3,3]
   - 1643.328303419059 R[3,4] - 1352.781167139699 R[3,5]
   - 936.4331100281483 R[3,6] - 581.0942725587191 R[3,7] = 0
 Constr_Left_0_diff_aux_3: Sold_0[3] - Q0_var[3] + Left_0_aux_3 = 0
 Constr_Lost_0_diff_aux_3: - Total_Demand_Up_to_K_minus_1[3] + Q0_var[3]
   + Lost_0_aux_3 = 0
 Constr_Q1_plus_left_3: - Left_0[3] + Q1_plus_left[3] - Q1_var[3] = 0
 Constr_Left_1_diff_aux_3: Sold_1[3] - Q1_plus_left[3] + Left_0_aux_3 = 0
 Constr_Lost_1_diff_aux_3: - Total_Demand_from_k_to_T[3] + Q1_plus_left[3]
   + Lost_0_aux_3 = 0
 Profit_Constraint_3: - 500 Sold_0[3] + 500 Lost_0[3] - 500 Sold_1[3]
   + 500 Left_1[3] + 500 Lost_1[3] + profits_vars[3] = 0
 R84: 0.5916999328977518 alphas[0] + 0.8059439709717974 alphas[1]
   + 1.666851549198021 alphas[2] + 1.12460937732089 alphas[3] - alphas[4]
   + f_var[4] = 0
 Q0_upper_bound_4: - 2053.400566507637 Fraction_for_second_order_amount[4]
   + Q0_var[4] = 0
 tau_computation_4_0: - betas[0] + 0.5916999328977518 gammas[0]
   + 0.8059439709717974 gammas[1] + 1.666851549198021 gammas[2]
   + 1.12460937732089 gammas[3] + tau[4,0] = 0
 tau_computation_4_1: - betas[1] + 0.5916999328977518 gammas[0]
   + 0.8059439709717974 gammas[1] + 1.666851549198021 gammas[2]
   + 1.12460937732089 gammas[3] + tau[4,1] = 0
 tau_computation_4_2: - betas[2] + 0.5916999328977518 gammas[0]
   + 0.8059439709717974 gammas[1] + 1.666851549198021 gammas[2]
   + 1.12460937732089 gammas[3] + tau[4,2] = 0
 tau_computation_4_3: - betas[3] + 0.5916999328977518 gammas[0]
   + 0.8059439709717974 gammas[1] + 1.666851549198021 gammas[2]
   + 1.12460937732089 gammas[3] + tau[4,3] = 0
 tau_computation_4_4: - betas[4] + 0.5916999328977518 gammas[0]
   + 0.8059439709717974 gammas[1] + 1.666851549198021 gammas[2]
   + 1.12460937732089 gammas[3] + tau[4,4] = 0
 tau_computation_4_5: - betas[5] + 0.5916999328977518 gammas[0]
   + 0.8059439709717974 gammas[1] + 1.666851549198021 gammas[2]
   + 1.12460937732089 gammas[3] + tau[4,5] = 0
 tau_computation_4_6: - betas[6] + 0.5916999328977518 gammas[0]
   + 0.8059439709717974 gammas[1] + 1.666851549198021 gammas[2]
   + 1.12460937732089 gammas[3] + tau[4,6] = 0
 tau_computation_4_7: - betas[7] + 0.5916999328977518 gammas[0]
   + 0.8059439709717974 gammas[1] + 1.666851549198021 gammas[2]
   + 1.12460937732089 gammas[3] + tau[4,7] = 0
 Ensure_only_one_R_true_4: R[4,0] + R[4,1] + R[4,2] + R[4,3] + R[4,4]
   + R[4,5] + R[4,6] + R[4,7] = 1
 Define_Q_hat_4: - 2144.595108668847 R[4,0] - 1183.894169975475 R[4,1]
   - 1320.219615932277 R[4,2] - 1407.758992777481 R[4,3]
   - 1583.22261240164 R[4,4] - 1583.223052404767 R[4,5]
   - 1559.209344238242 R[4,6] - 1587.52112455324 R[4,7] + Q_hat[4] = 0
 Adjust_Q_hat_4: Q0_var[4] - Q_hat[4] + Q_hat_adjusted[4] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_4: Total_Demand_Up_to_K_minus_1[4]
   - 264.4641065267176 R[4,0] - 376.6591372109277 R[4,1]
   - 702.2442772072504 R[4,2] - 814.43256281288 R[4,3]
   - 1030.521272496387 R[4,4] - 1142.709558376377 R[4,5]
   - 1169.5467268745 R[4,6] - 1363.144551235566 R[4,7] = 0
 Constr_Total_Demand_from_K_to_T_4: Total_Demand_from_k_to_T[4]
   - 1323.057016468828 R[4,0] - 1210.861985784617 R[4,1]
   - 885.2768457882947 R[4,2] - 773.0885601826651 R[4,3]
   - 556.9998504991578 R[4,4] - 444.8115646191683 R[4,5]
   - 417.9743961210451 R[4,6] - 224.376571759979 R[4,7] = 0
 Constr_Left_0_diff_aux_4: Sold_0[4] - Q0_var[4] + Left_0_aux_4 = 0
 Constr_Lost_0_diff_aux_4: - Total_Demand_Up_to_K_minus_1[4] + Q0_var[4]
   + Lost_0_aux_4 = 0
 Constr_Q1_plus_left_4: - Left_0[4] + Q1_plus_left[4] - Q1_var[4] = 0
 Constr_Left_1_diff_aux_4: Sold_1[4] - Q1_plus_left[4] + Left_0_aux_4 = 0
 Constr_Lost_1_diff_aux_4: - Total_Demand_from_k_to_T[4] + Q1_plus_left[4]
   + Lost_0_aux_4 = 0
 Profit_Constraint_4: - 500 Sold_0[4] + 500 Lost_0[4] - 500 Sold_1[4]
   + 500 Left_1[4] + 500 Lost_1[4] + profits_vars[4] = 0
 R105: - 0.1951939812905471 alphas[0] + 0.9627548780597103 alphas[1]
   + 0.2122303512187826 alphas[2] - 1.148715514237678 alphas[3] - alphas[4]
   + f_var[5] = 0
 Q0_upper_bound_5: - 2053.400566507637 Fraction_for_second_order_amount[5]
   + Q0_var[5] = 0
 tau_computation_5_0: - betas[0] - 0.1951939812905471 gammas[0]
   + 0.9627548780597103 gammas[1] + 0.2122303512187826 gammas[2]
   - 1.148715514237678 gammas[3] + tau[5,0] = 0
 tau_computation_5_1: - betas[1] - 0.1951939812905471 gammas[0]
   + 0.9627548780597103 gammas[1] + 0.2122303512187826 gammas[2]
   - 1.148715514237678 gammas[3] + tau[5,1] = 0
 tau_computation_5_2: - betas[2] - 0.1951939812905471 gammas[0]
   + 0.9627548780597103 gammas[1] + 0.2122303512187826 gammas[2]
   - 1.148715514237678 gammas[3] + tau[5,2] = 0
 tau_computation_5_3: - betas[3] - 0.1951939812905471 gammas[0]
   + 0.9627548780597103 gammas[1] + 0.2122303512187826 gammas[2]
   - 1.148715514237678 gammas[3] + tau[5,3] = 0
 tau_computation_5_4: - betas[4] - 0.1951939812905471 gammas[0]
   + 0.9627548780597103 gammas[1] + 0.2122303512187826 gammas[2]
   - 1.148715514237678 gammas[3] + tau[5,4] = 0
 tau_computation_5_5: - betas[5] - 0.1951939812905471 gammas[0]
   + 0.9627548780597103 gammas[1] + 0.2122303512187826 gammas[2]
   - 1.148715514237678 gammas[3] + tau[5,5] = 0
 tau_computation_5_6: - betas[6] - 0.1951939812905471 gammas[0]
   + 0.9627548780597103 gammas[1] + 0.2122303512187826 gammas[2]
   - 1.148715514237678 gammas[3] + tau[5,6] = 0
 tau_computation_5_7: - betas[7] - 0.1951939812905471 gammas[0]
   + 0.9627548780597103 gammas[1] + 0.2122303512187826 gammas[2]
   - 1.148715514237678 gammas[3] + tau[5,7] = 0
 Ensure_only_one_R_true_5: R[5,0] + R[5,1] + R[5,2] + R[5,3] + R[5,4]
   + R[5,5] + R[5,6] + R[5,7] = 1
 Define_Q_hat_5: - 2175.820552811292 R[5,0] - 1951.22763816568 R[5,1]
   - 1960.339006372115 R[5,2] - 1970.619949369073 R[5,3]
   - 1878.954140002629 R[5,4] - 1878.954233111025 R[5,5]
   - 1936.033769510368 R[5,6] - 1955.751423474789 R[5,7] + Q_hat[5] = 0
 Adjust_Q_hat_5: Q0_var[5] - Q_hat[5] + Q_hat_adjusted[5] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_5: Total_Demand_Up_to_K_minus_1[5]
   - 276.4070234549736 R[5,0] - 470.44636046504 R[5,1]
   - 723.9445597797467 R[5,2] - 917.9838933489232 R[5,3]
   - 973.8586516092186 R[5,4] - 1167.897985213678 R[5,5]
   - 1380.399153393131 R[5,6] - 1567.672756172823 R[5,7] = 0
 Constr_Total_Demand_from_K_to_T_5: Total_Demand_from_k_to_T[5]
   - 1679.344399926768 R[5,0] - 1485.305062916701 R[5,1]
   - 1231.806863601995 R[5,2] - 1037.767530032818 R[5,3]
   - 981.8927717725228 R[5,4] - 787.8534381680636 R[5,5]
   - 575.3522699886098 R[5,6] - 388.0786672089184 R[5,7] = 0
 Constr_Left_0_diff_aux_5: Sold_0[5] - Q0_var[5] + Left_0_aux_5 = 0
 Constr_Lost_0_diff_aux_5: - Total_Demand_Up_to_K_minus_1[5] + Q0_var[5]
   + Lost_0_aux_5 = 0
 Constr_Q1_plus_left_5: - Left_0[5] + Q1_plus_left[5] - Q1_var[5] = 0
 Constr_Left_1_diff_aux_5: Sold_1[5] - Q1_plus_left[5] + Left_0_aux_5 = 0
 Constr_Lost_1_diff_aux_5: - Total_Demand_from_k_to_T[5] + Q1_plus_left[5]
   + Lost_0_aux_5 = 0
 Profit_Constraint_5: - 500 Sold_0[5] + 500 Lost_0[5] - 500 Sold_1[5]
   + 500 Left_1[5] + 500 Lost_1[5] + profits_vars[5] = 0
 R126: 0.5423687413596802 alphas[0] - 0.6585168211974903 alphas[1]
   + 0.1416568156698964 alphas[2] - 1.020986388394283 alphas[3] - alphas[4]
   + f_var[6] = 0
 Q0_upper_bound_6: - 2053.400566507637 Fraction_for_second_order_amount[6]
   + Q0_var[6] = 0
 tau_computation_6_0: - betas[0] + 0.5423687413596802 gammas[0]
   - 0.6585168211974903 gammas[1] + 0.1416568156698964 gammas[2]
   - 1.020986388394283 gammas[3] + tau[6,0] = 0
 tau_computation_6_1: - betas[1] + 0.5423687413596802 gammas[0]
   - 0.6585168211974903 gammas[1] + 0.1416568156698964 gammas[2]
   - 1.020986388394283 gammas[3] + tau[6,1] = 0
 tau_computation_6_2: - betas[2] + 0.5423687413596802 gammas[0]
   - 0.6585168211974903 gammas[1] + 0.1416568156698964 gammas[2]
   - 1.020986388394283 gammas[3] + tau[6,2] = 0
 tau_computation_6_3: - betas[3] + 0.5423687413596802 gammas[0]
   - 0.6585168211974903 gammas[1] + 0.1416568156698964 gammas[2]
   - 1.020986388394283 gammas[3] + tau[6,3] = 0
 tau_computation_6_4: - betas[4] + 0.5423687413596802 gammas[0]
   - 0.6585168211974903 gammas[1] + 0.1416568156698964 gammas[2]
   - 1.020986388394283 gammas[3] + tau[6,4] = 0
 tau_computation_6_5: - betas[5] + 0.5423687413596802 gammas[0]
   - 0.6585168211974903 gammas[1] + 0.1416568156698964 gammas[2]
   - 1.020986388394283 gammas[3] + tau[6,5] = 0
 tau_computation_6_6: - betas[6] + 0.5423687413596802 gammas[0]
   - 0.6585168211974903 gammas[1] + 0.1416568156698964 gammas[2]
   - 1.020986388394283 gammas[3] + tau[6,6] = 0
 tau_computation_6_7: - betas[7] + 0.5423687413596802 gammas[0]
   - 0.6585168211974903 gammas[1] + 0.1416568156698964 gammas[2]
   - 1.020986388394283 gammas[3] + tau[6,7] = 0
 Ensure_only_one_R_true_6: R[6,0] + R[6,1] + R[6,2] + R[6,3] + R[6,4]
   + R[6,5] + R[6,6] + R[6,7] = 1
 Define_Q_hat_6: - 2120.286889188916 R[6,0] - 1984.063411823704 R[6,1]
   - 2017.133430239564 R[6,2] - 2021.949684817188 R[6,3]
   - 1918.97125461595 R[6,4] - 1918.971835293632 R[6,5]
   - 1967.459232064394 R[6,6] - 2068.548967509328 R[6,7] + Q_hat[6] = 0
 Adjust_Q_hat_6: Q0_var[6] - Q_hat[6] + Q_hat_adjusted[6] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_6: Total_Demand_Up_to_K_minus_1[6]
   - 255.1668473877191 R[6,0] - 453.1773501135664 R[6,1]
   - 712.1907199647312 R[6,2] - 910.2012031059226 R[6,3]
   - 972.0292317124544 R[6,4] - 1170.039714530414 R[6,5]
   - 1384.282485519538 R[6,6] - 1672.528001503375 R[6,7] = 0
 Constr_Total_Demand_from_K_to_T_6: Total_Demand_from_k_to_T[6]
   - 1813.382119751576 R[6,0] - 1615.371617025728 R[6,1]
   - 1356.358247174564 R[6,2] - 1158.347764033372 R[6,3]
   - 1096.519735426841 R[6,4] - 898.509252608881 R[6,5]
   - 684.2664816197569 R[6,6] - 396.0209656359195 R[6,7] = 0
 Constr_Left_0_diff_aux_6: Sold_0[6] - Q0_var[6] + Left_0_aux_6 = 0
 Constr_Lost_0_diff_aux_6: - Total_Demand_Up_to_K_minus_1[6] + Q0_var[6]
   + Lost_0_aux_6 = 0
 Constr_Q1_plus_left_6: - Left_0[6] + Q1_plus_left[6] - Q1_var[6] = 0
 Constr_Left_1_diff_aux_6: Sold_1[6] - Q1_plus_left[6] + Left_0_aux_6 = 0
 Constr_Lost_1_diff_aux_6: - Total_Demand_from_k_to_T[6] + Q1_plus_left[6]
   + Lost_0_aux_6 = 0
 Profit_Constraint_6: - 500 Sold_0[6] + 500 Lost_0[6] - 500 Sold_1[6]
   + 500 Left_1[6] + 500 Lost_1[6] + profits_vars[6] = 0
 R147: - 1.065789723894727 alphas[0] + 0.6595252882541187 alphas[1]
   + 0.2295817456905412 alphas[2] - 0.2731871634863502 alphas[3]
   - alphas[4] + f_var[7] = 0
 Q0_upper_bound_7: - 2053.400566507637 Fraction_for_second_order_amount[7]
   + Q0_var[7] = 0
 tau_computation_7_0: - betas[0] - 1.065789723894727 gammas[0]
   + 0.6595252882541187 gammas[1] + 0.2295817456905412 gammas[2]
   - 0.2731871634863502 gammas[3] + tau[7,0] = 0
 tau_computation_7_1: - betas[1] - 1.065789723894727 gammas[0]
   + 0.6595252882541187 gammas[1] + 0.2295817456905412 gammas[2]
   - 0.2731871634863502 gammas[3] + tau[7,1] = 0
 tau_computation_7_2: - betas[2] - 1.065789723894727 gammas[0]
   + 0.6595252882541187 gammas[1] + 0.2295817456905412 gammas[2]
   - 0.2731871634863502 gammas[3] + tau[7,2] = 0
 tau_computation_7_3: - betas[3] - 1.065789723894727 gammas[0]
   + 0.6595252882541187 gammas[1] + 0.2295817456905412 gammas[2]
   - 0.2731871634863502 gammas[3] + tau[7,3] = 0
 tau_computation_7_4: - betas[4] - 1.065789723894727 gammas[0]
   + 0.6595252882541187 gammas[1] + 0.2295817456905412 gammas[2]
   - 0.2731871634863502 gammas[3] + tau[7,4] = 0
 tau_computation_7_5: - betas[5] - 1.065789723894727 gammas[0]
   + 0.6595252882541187 gammas[1] + 0.2295817456905412 gammas[2]
   - 0.2731871634863502 gammas[3] + tau[7,5] = 0
 tau_computation_7_6: - betas[6] - 1.065789723894727 gammas[0]
   + 0.6595252882541187 gammas[1] + 0.2295817456905412 gammas[2]
   - 0.2731871634863502 gammas[3] + tau[7,6] = 0
 tau_computation_7_7: - betas[7] - 1.065789723894727 gammas[0]
   + 0.6595252882541187 gammas[1] + 0.2295817456905412 gammas[2]
   - 0.2731871634863502 gammas[3] + tau[7,7] = 0
 Ensure_only_one_R_true_7: R[7,0] + R[7,1] + R[7,2] + R[7,3] + R[7,4]
   + R[7,5] + R[7,6] + R[7,7] = 1
 Define_Q_hat_7: - 1632.635076428851 R[7,0] - 1900.246313651594 R[7,1]
   - 1870.755961928724 R[7,2] - 1885.087350226062 R[7,3]
   - 1716.58681088837 R[7,4] - 1716.587298707414 R[7,5]
   - 1643.710431062219 R[7,6] - 1665.617778576442 R[7,7] + Q_hat[7] = 0
 Adjust_Q_hat_7: Q0_var[7] - Q_hat[7] + Q_hat_adjusted[7] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_7: Total_Demand_Up_to_K_minus_1[7]
   - 68.65276409273326 R[7,0] - 261.7157423113841 R[7,1]
   - 338.0468765949859 R[7,2] - 531.1098529444025 R[7,3]
   - 677.3619666273681 R[7,4] - 870.4249430750865 R[7,5]
   - 1076.330741541827 R[7,6] - 1279.491825681141 R[7,7] = 0
 Constr_Total_Demand_from_K_to_T_7: Total_Demand_from_k_to_T[7]
   - 1596.965014483844 R[7,0] - 1403.902036265193 R[7,1]
   - 1327.570901981592 R[7,2] - 1134.507925632175 R[7,3]
   - 988.2558119492095 R[7,4] - 795.1928355014911 R[7,5]
   - 589.2870370347504 R[7,6] - 386.1259528954367 R[7,7] = 0
 Constr_Left_0_diff_aux_7: Sold_0[7] - Q0_var[7] + Left_0_aux_7 = 0
 Constr_Lost_0_diff_aux_7: - Total_Demand_Up_to_K_minus_1[7] + Q0_var[7]
   + Lost_0_aux_7 = 0
 Constr_Q1_plus_left_7: - Left_0[7] + Q1_plus_left[7] - Q1_var[7] = 0
 Constr_Left_1_diff_aux_7: Sold_1[7] - Q1_plus_left[7] + Left_0_aux_7 = 0
 Constr_Lost_1_diff_aux_7: - Total_Demand_from_k_to_T[7] + Q1_plus_left[7]
   + Lost_0_aux_7 = 0
 Profit_Constraint_7: - 500 Sold_0[7] + 500 Lost_0[7] - 500 Sold_1[7]
   + 500 Left_1[7] + 500 Lost_1[7] + profits_vars[7] = 0
 R168: - 1.320333411356792 alphas[0] - 0.0427170655502229 alphas[1]
   + 1.780907278927179 alphas[2] + 0.2350935931801337 alphas[3] - alphas[4]
   + f_var[8] = 0
 Q0_upper_bound_8: - 2053.400566507637 Fraction_for_second_order_amount[8]
   + Q0_var[8] = 0
 tau_computation_8_0: - betas[0] - 1.320333411356792 gammas[0]
   - 0.0427170655502229 gammas[1] + 1.780907278927179 gammas[2]
   + 0.2350935931801337 gammas[3] + tau[8,0] = 0
 tau_computation_8_1: - betas[1] - 1.320333411356792 gammas[0]
   - 0.0427170655502229 gammas[1] + 1.780907278927179 gammas[2]
   + 0.2350935931801337 gammas[3] + tau[8,1] = 0
 tau_computation_8_2: - betas[2] - 1.320333411356792 gammas[0]
   - 0.0427170655502229 gammas[1] + 1.780907278927179 gammas[2]
   + 0.2350935931801337 gammas[3] + tau[8,2] = 0
 tau_computation_8_3: - betas[3] - 1.320333411356792 gammas[0]
   - 0.0427170655502229 gammas[1] + 1.780907278927179 gammas[2]
   + 0.2350935931801337 gammas[3] + tau[8,3] = 0
 tau_computation_8_4: - betas[4] - 1.320333411356792 gammas[0]
   - 0.0427170655502229 gammas[1] + 1.780907278927179 gammas[2]
   + 0.2350935931801337 gammas[3] + tau[8,4] = 0
 tau_computation_8_5: - betas[5] - 1.320333411356792 gammas[0]
   - 0.0427170655502229 gammas[1] + 1.780907278927179 gammas[2]
   + 0.2350935931801337 gammas[3] + tau[8,5] = 0
 tau_computation_8_6: - betas[6] - 1.320333411356792 gammas[0]
   - 0.0427170655502229 gammas[1] + 1.780907278927179 gammas[2]
   + 0.2350935931801337 gammas[3] + tau[8,6] = 0
 tau_computation_8_7: - betas[7] - 1.320333411356792 gammas[0]
   - 0.0427170655502229 gammas[1] + 1.780907278927179 gammas[2]
   + 0.2350935931801337 gammas[3] + tau[8,7] = 0
 Ensure_only_one_R_true_8: R[8,0] + R[8,1] + R[8,2] + R[8,3] + R[8,4]
   + R[8,5] + R[8,6] + R[8,7] = 1
 Define_Q_hat_8: - 1971.712159038372 R[8,0] - 1110.34123938017 R[8,1]
   - 1019.181074280429 R[8,2] - 839.5362731492149 R[8,3]
   - 906.8512823132319 R[8,4] - 906.8479778313489 R[8,5]
   - 899.4198071667295 R[8,6] - 920.8038788696135 R[8,7] + Q_hat[8] = 0
 Adjust_Q_hat_8: Q0_var[8] - Q_hat[8] + Q_hat_adjusted[8] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_8: Total_Demand_Up_to_K_minus_1[8]
   - 198.3408931520826 R[8,0] - 304.0914239454083 R[8,1]
   - 397.9283659900643 R[8,2] - 503.6987738674777 R[8,3]
   - 503.6987738674777 R[8,4] - 609.4691820276324 R[8,5]
   - 671.6016029470612 R[8,6] - 709.2630595513233 R[8,7] = 0
 Constr_Total_Demand_from_K_to_T_8: Total_Demand_from_k_to_T[8]
   - 722.4629827195499 R[8,0] - 616.7124519262242 R[8,1]
   - 522.8755098815682 R[8,2] - 417.1051020041548 R[8,3]
   - 417.1051020041548 R[8,4] - 311.3346938440001 R[8,5]
   - 249.2022729245712 R[8,6] - 211.5408163203091 R[8,7] = 0
 Constr_Left_0_diff_aux_8: Sold_0[8] - Q0_var[8] + Left_0_aux_8 = 0
 Constr_Lost_0_diff_aux_8: - Total_Demand_Up_to_K_minus_1[8] + Q0_var[8]
   + Lost_0_aux_8 = 0
 Constr_Q1_plus_left_8: - Left_0[8] + Q1_plus_left[8] - Q1_var[8] = 0
 Constr_Left_1_diff_aux_8: Sold_1[8] - Q1_plus_left[8] + Left_0_aux_8 = 0
 Constr_Lost_1_diff_aux_8: - Total_Demand_from_k_to_T[8] + Q1_plus_left[8]
   + Lost_0_aux_8 = 0
 Profit_Constraint_8: - 500 Sold_0[8] + 500 Lost_0[8] - 500 Sold_1[8]
   + 500 Left_1[8] + 500 Lost_1[8] + profits_vars[8] = 0
 R189: 0.7340851174744587 alphas[0] + 0.6887531183124831 alphas[1]
   - 1.460925496263445 alphas[2] + 1.29617249954311 alphas[3] - alphas[4]
   + f_var[9] = 0
 Q0_upper_bound_9: - 2053.400566507637 Fraction_for_second_order_amount[9]
   + Q0_var[9] = 0
 tau_computation_9_0: - betas[0] + 0.7340851174744587 gammas[0]
   + 0.6887531183124831 gammas[1] - 1.460925496263445 gammas[2]
   + 1.29617249954311 gammas[3] + tau[9,0] = 0
 tau_computation_9_1: - betas[1] + 0.7340851174744587 gammas[0]
   + 0.6887531183124831 gammas[1] - 1.460925496263445 gammas[2]
   + 1.29617249954311 gammas[3] + tau[9,1] = 0
 tau_computation_9_2: - betas[2] + 0.7340851174744587 gammas[0]
   + 0.6887531183124831 gammas[1] - 1.460925496263445 gammas[2]
   + 1.29617249954311 gammas[3] + tau[9,2] = 0
 tau_computation_9_3: - betas[3] + 0.7340851174744587 gammas[0]
   + 0.6887531183124831 gammas[1] - 1.460925496263445 gammas[2]
   + 1.29617249954311 gammas[3] + tau[9,3] = 0
 tau_computation_9_4: - betas[4] + 0.7340851174744587 gammas[0]
   + 0.6887531183124831 gammas[1] - 1.460925496263445 gammas[2]
   + 1.29617249954311 gammas[3] + tau[9,4] = 0
 tau_computation_9_5: - betas[5] + 0.7340851174744587 gammas[0]
   + 0.6887531183124831 gammas[1] - 1.460925496263445 gammas[2]
   + 1.29617249954311 gammas[3] + tau[9,5] = 0
 tau_computation_9_6: - betas[6] + 0.7340851174744587 gammas[0]
   + 0.6887531183124831 gammas[1] - 1.460925496263445 gammas[2]
   + 1.29617249954311 gammas[3] + tau[9,6] = 0
 tau_computation_9_7: - betas[7] + 0.7340851174744587 gammas[0]
   + 0.6887531183124831 gammas[1] - 1.460925496263445 gammas[2]
   + 1.29617249954311 gammas[3] + tau[9,7] = 0
 Ensure_only_one_R_true_9: R[9,0] + R[9,1] + R[9,2] + R[9,3] + R[9,4]
   + R[9,5] + R[9,6] + R[9,7] = 1
 Define_Q_hat_9: - 2411.05594373497 R[9,0] - 2849.272567420796 R[9,1]
   - 2771.912502594555 R[9,2] - 2773.59578664033 R[9,3]
   - 2851.639396660061 R[9,4] - 2851.63904252772 R[9,5]
   - 2870.553405907306 R[9,6] - 2802.887150206102 R[9,7] + Q_hat[9] = 0
 Adjust_Q_hat_9: Q0_var[9] - Q_hat[9] + Q_hat_adjusted[9] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_9: Total_Demand_Up_to_K_minus_1[9]
   - 366.3784172827553 R[9,0] - 654.5656697364941 R[9,1]
   - 926.940265867065 R[9,2] - 1215.127518735543 R[9,3]
   - 1505.4219260244 R[9,4] - 1793.609178903646 R[9,5]
   - 2081.710739351498 R[9,6] - 2226.5126428821 R[9,7] = 0
 Constr_Total_Demand_from_K_to_T_9: Total_Demand_from_k_to_T[9]
   - 2436.508731357838 R[9,0] - 2148.321478904099 R[9,1]
   - 1875.946882773527 R[9,2] - 1587.75962990505 R[9,3]
   - 1297.465222616193 R[9,4] - 1009.277969736947 R[9,5]
   - 721.1764092890943 R[9,6] - 576.3745057584925 R[9,7] = 0
 Constr_Left_0_diff_aux_9: Sold_0[9] - Q0_var[9] + Left_0_aux_9 = 0
 Constr_Lost_0_diff_aux_9: - Total_Demand_Up_to_K_minus_1[9] + Q0_var[9]
   + Lost_0_aux_9 = 0
 Constr_Q1_plus_left_9: - Left_0[9] + Q1_plus_left[9] - Q1_var[9] = 0
 Constr_Left_1_diff_aux_9: Sold_1[9] - Q1_plus_left[9] + Left_0_aux_9 = 0
 Constr_Lost_1_diff_aux_9: - Total_Demand_from_k_to_T[9] + Q1_plus_left[9]
   + Lost_0_aux_9 = 0
 Profit_Constraint_9: - 500 Sold_0[9] + 500 Lost_0[9] - 500 Sold_1[9]
   + 500 Left_1[9] + 500 Lost_1[9] + profits_vars[9] = 0
 R210: - 0.7115448711071122 alphas[0] + 0.9704687421482601 alphas[1]
   + 0.7855052919242557 alphas[2] - 0.6760262272279475 alphas[3]
   - alphas[4] + f_var[10] = 0
 Q0_upper_bound_10:
   - 2053.400566507637 Fraction_for_second_order_amount[10] + Q0_var[10]
   = 0
 tau_computation_10_0: - betas[0] - 0.7115448711071122 gammas[0]
   + 0.9704687421482601 gammas[1] + 0.7855052919242557 gammas[2]
   - 0.6760262272279475 gammas[3] + tau[10,0] = 0
 tau_computation_10_1: - betas[1] - 0.7115448711071122 gammas[0]
   + 0.9704687421482601 gammas[1] + 0.7855052919242557 gammas[2]
   - 0.6760262272279475 gammas[3] + tau[10,1] = 0
 tau_computation_10_2: - betas[2] - 0.7115448711071122 gammas[0]
   + 0.9704687421482601 gammas[1] + 0.7855052919242557 gammas[2]
   - 0.6760262272279475 gammas[3] + tau[10,2] = 0
 tau_computation_10_3: - betas[3] - 0.7115448711071122 gammas[0]
   + 0.9704687421482601 gammas[1] + 0.7855052919242557 gammas[2]
   - 0.6760262272279475 gammas[3] + tau[10,3] = 0
 tau_computation_10_4: - betas[4] - 0.7115448711071122 gammas[0]
   + 0.9704687421482601 gammas[1] + 0.7855052919242557 gammas[2]
   - 0.6760262272279475 gammas[3] + tau[10,4] = 0
 tau_computation_10_5: - betas[5] - 0.7115448711071122 gammas[0]
   + 0.9704687421482601 gammas[1] + 0.7855052919242557 gammas[2]
   - 0.6760262272279475 gammas[3] + tau[10,5] = 0
 tau_computation_10_6: - betas[6] - 0.7115448711071122 gammas[0]
   + 0.9704687421482601 gammas[1] + 0.7855052919242557 gammas[2]
   - 0.6760262272279475 gammas[3] + tau[10,6] = 0
 tau_computation_10_7: - betas[7] - 0.7115448711071122 gammas[0]
   + 0.9704687421482601 gammas[1] + 0.7855052919242557 gammas[2]
   - 0.6760262272279475 gammas[3] + tau[10,7] = 0
 Ensure_only_one_R_true_10: R[10,0] + R[10,1] + R[10,2] + R[10,3] + R[10,4]
   + R[10,5] + R[10,6] + R[10,7] = 1
 Define_Q_hat_10: - 1575.88419069516 R[10,0] - 1603.509020082587 R[10,1]
   - 1669.37256236232 R[10,2] - 1672.156094423305 R[10,3]
   - 1856.903318963589 R[10,4] - 1856.903026175613 R[10,5]
   - 1915.494648159011 R[10,6] - 2013.535923713845 R[10,7] + Q_hat[10] = 0
 Adjust_Q_hat_10: Q0_var[10] - Q_hat[10] + Q_hat_adjusted[10] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_10: Total_Demand_Up_to_K_minus_1[10]
   - 46.94703235580273 R[10,0] - 208.7281512687877 R[10,1]
   - 338.3052986318435 R[10,2] - 500.0866431544503 R[10,3]
   - 913.1750791073852 R[10,4] - 1074.956423761792 R[10,5]
   - 1347.936907398919 R[10,6] - 1689.973234955106 R[10,7] = 0
 Constr_Total_Demand_from_K_to_T_10: Total_Demand_from_k_to_T[10]
   - 1966.588891908117 R[10,0] - 1804.807772995132 R[10,1]
   - 1675.230625632076 R[10,2] - 1513.449281109469 R[10,3]
   - 1100.360845156534 R[10,4] - 938.5795005021275 R[10,5]
   - 665.5990168650003 R[10,6] - 323.5626893088137 R[10,7] = 0
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
 R231: 0.2190709461845003 alphas[0] + 1.130665947024758 alphas[1]
   - 0.9884669399077892 alphas[2] + 0.0895652096157003 alphas[3]
   - alphas[4] + f_var[11] = 0
 Q0_upper_bound_11:
   - 2053.400566507637 Fraction_for_second_order_amount[11] + Q0_var[11]
   = 0
 tau_computation_11_0: - betas[0] + 0.2190709461845003 gammas[0]
   + 1.130665947024758 gammas[1] - 0.9884669399077892 gammas[2]
   + 0.0895652096157003 gammas[3] + tau[11,0] = 0
 tau_computation_11_1: - betas[1] + 0.2190709461845003 gammas[0]
   + 1.130665947024758 gammas[1] - 0.9884669399077892 gammas[2]
   + 0.0895652096157003 gammas[3] + tau[11,1] = 0
 tau_computation_11_2: - betas[2] + 0.2190709461845003 gammas[0]
   + 1.130665947024758 gammas[1] - 0.9884669399077892 gammas[2]
   + 0.0895652096157003 gammas[3] + tau[11,2] = 0
 tau_computation_11_3: - betas[3] + 0.2190709461845003 gammas[0]
   + 1.130665947024758 gammas[1] - 0.9884669399077892 gammas[2]
   + 0.0895652096157003 gammas[3] + tau[11,3] = 0
 tau_computation_11_4: - betas[4] + 0.2190709461845003 gammas[0]
   + 1.130665947024758 gammas[1] - 0.9884669399077892 gammas[2]
   + 0.0895652096157003 gammas[3] + tau[11,4] = 0
 tau_computation_11_5: - betas[5] + 0.2190709461845003 gammas[0]
   + 1.130665947024758 gammas[1] - 0.9884669399077892 gammas[2]
   + 0.0895652096157003 gammas[3] + tau[11,5] = 0
 tau_computation_11_6: - betas[6] + 0.2190709461845003 gammas[0]
   + 1.130665947024758 gammas[1] - 0.9884669399077892 gammas[2]
   + 0.0895652096157003 gammas[3] + tau[11,6] = 0
 tau_computation_11_7: - betas[7] + 0.2190709461845003 gammas[0]
   + 1.130665947024758 gammas[1] - 0.9884669399077892 gammas[2]
   + 0.0895652096157003 gammas[3] + tau[11,7] = 0
 Ensure_only_one_R_true_11: R[11,0] + R[11,1] + R[11,2] + R[11,3] + R[11,4]
   + R[11,5] + R[11,6] + R[11,7] = 1
 Define_Q_hat_11: - 2048.572436004985 R[11,0] - 2572.873144509696 R[11,1]
   - 2437.250474753614 R[11,2] - 2454.407489072059 R[11,3]
   - 2697.799050765736 R[11,4] - 2697.799619589837 R[11,5]
   - 2822.267274891037 R[11,6] - 2831.667027060258 R[11,7] + Q_hat[11] = 0
 Adjust_Q_hat_11: Q0_var[11] - Q_hat[11] + Q_hat_adjusted[11] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_11: Total_Demand_Up_to_K_minus_1[11]
   - 227.7379418764355 R[11,0] - 489.3401098607683 R[11,1]
   - 610.2132025604587 R[11,2] - 871.8153712133555 R[11,3]
   - 1387.436949878543 R[11,4] - 1649.039117996325 R[11,5]
   - 2079.056035025463 R[11,6] - 2308.46269322693 R[11,7] = 0
 Constr_Total_Demand_from_K_to_T_11: Total_Demand_from_k_to_T[11]
   - 2603.929087586059 R[11,0] - 2342.326919601726 R[11,1]
   - 2221.453826902035 R[11,2] - 1959.851658249138 R[11,3]
   - 1444.230079583951 R[11,4] - 1182.627911466169 R[11,5]
   - 752.6109944370311 R[11,6] - 523.2043362355639 R[11,7] = 0
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
 R252: 0.0804518423132196 alphas[0] - 0.6691360908629208 alphas[1]
   - 0.3356756145437382 alphas[2] - 0.7533331671833704 alphas[3]
   - alphas[4] + f_var[12] = 0
 Q0_upper_bound_12:
   - 2053.400566507637 Fraction_for_second_order_amount[12] + Q0_var[12]
   = 0
 tau_computation_12_0: - betas[0] + 0.0804518423132196 gammas[0]
   - 0.6691360908629208 gammas[1] - 0.3356756145437382 gammas[2]
   - 0.7533331671833704 gammas[3] + tau[12,0] = 0
 tau_computation_12_1: - betas[1] + 0.0804518423132196 gammas[0]
   - 0.6691360908629208 gammas[1] - 0.3356756145437382 gammas[2]
   - 0.7533331671833704 gammas[3] + tau[12,1] = 0
 tau_computation_12_2: - betas[2] + 0.0804518423132196 gammas[0]
   - 0.6691360908629208 gammas[1] - 0.3356756145437382 gammas[2]
   - 0.7533331671833704 gammas[3] + tau[12,2] = 0
 tau_computation_12_3: - betas[3] + 0.0804518423132196 gammas[0]
   - 0.6691360908629208 gammas[1] - 0.3356756145437382 gammas[2]
   - 0.7533331671833704 gammas[3] + tau[12,3] = 0
 tau_computation_12_4: - betas[4] + 0.0804518423132196 gammas[0]
   - 0.6691360908629208 gammas[1] - 0.3356756145437382 gammas[2]
   - 0.7533331671833704 gammas[3] + tau[12,4] = 0
 tau_computation_12_5: - betas[5] + 0.0804518423132196 gammas[0]
   - 0.6691360908629208 gammas[1] - 0.3356756145437382 gammas[2]
   - 0.7533331671833704 gammas[3] + tau[12,5] = 0
 tau_computation_12_6: - betas[6] + 0.0804518423132196 gammas[0]
   - 0.6691360908629208 gammas[1] - 0.3356756145437382 gammas[2]
   - 0.7533331671833704 gammas[3] + tau[12,6] = 0
 tau_computation_12_7: - betas[7] + 0.0804518423132196 gammas[0]
   - 0.6691360908629208 gammas[1] - 0.3356756145437382 gammas[2]
   - 0.7533331671833704 gammas[3] + tau[12,7] = 0
 Ensure_only_one_R_true_12: R[12,0] + R[12,1] + R[12,2] + R[12,3] + R[12,4]
   + R[12,5] + R[12,6] + R[12,7] = 1
 Define_Q_hat_12: - 1862.706894967045 R[12,0] - 2215.245708448073 R[12,1]
   - 2332.277850525377 R[12,2] - 2313.224099498304 R[12,3]
   - 2434.578369476011 R[12,4] - 2434.577867996014 R[12,5]
   - 2342.663489629992 R[12,6] - 2325.669687732558 R[12,7] + Q_hat[12] = 0
 Adjust_Q_hat_12: Q0_var[12] - Q_hat[12] + Q_hat_adjusted[12] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_12: Total_Demand_Up_to_K_minus_1[12]
   - 156.649226330381 R[12,0] - 381.5190448789675 R[12,1]
   - 646.4785447920035 R[12,2] - 871.3483635812364 R[12,3]
   - 1225.286226414859 R[12,4] - 1450.156045244235 R[12,5]
   - 1609.664354805742 R[12,6] - 1875.930050515312 R[12,7] = 0
 Constr_Total_Demand_from_K_to_T_12: Total_Demand_from_k_to_T[12]
   - 2169.020461843683 R[12,0] - 1944.150643295096 R[12,1]
   - 1679.191143382061 R[12,2] - 1454.321324592828 R[12,3]
   - 1100.383461759205 R[12,4] - 875.5136429298292 R[12,5]
   - 716.0053333683218 R[12,6] - 449.7396376587521 R[12,7] = 0
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
 R273: - 1.185550778167024 alphas[0] + 1.03899428175183 alphas[1]
   + 1.450554690095896 alphas[2] - 1.206394289581885 alphas[3] - alphas[4]
   + f_var[13] = 0
 Q0_upper_bound_13:
   - 2053.400566507637 Fraction_for_second_order_amount[13] + Q0_var[13]
   = 0
 tau_computation_13_0: - betas[0] - 1.185550778167024 gammas[0]
   + 1.03899428175183 gammas[1] + 1.450554690095896 gammas[2]
   - 1.206394289581885 gammas[3] + tau[13,0] = 0
 tau_computation_13_1: - betas[1] - 1.185550778167024 gammas[0]
   + 1.03899428175183 gammas[1] + 1.450554690095896 gammas[2]
   - 1.206394289581885 gammas[3] + tau[13,1] = 0
 tau_computation_13_2: - betas[2] - 1.185550778167024 gammas[0]
   + 1.03899428175183 gammas[1] + 1.450554690095896 gammas[2]
   - 1.206394289581885 gammas[3] + tau[13,2] = 0
 tau_computation_13_3: - betas[3] - 1.185550778167024 gammas[0]
   + 1.03899428175183 gammas[1] + 1.450554690095896 gammas[2]
   - 1.206394289581885 gammas[3] + tau[13,3] = 0
 tau_computation_13_4: - betas[4] - 1.185550778167024 gammas[0]
   + 1.03899428175183 gammas[1] + 1.450554690095896 gammas[2]
   - 1.206394289581885 gammas[3] + tau[13,4] = 0
 tau_computation_13_5: - betas[5] - 1.185550778167024 gammas[0]
   + 1.03899428175183 gammas[1] + 1.450554690095896 gammas[2]
   - 1.206394289581885 gammas[3] + tau[13,5] = 0
 tau_computation_13_6: - betas[6] - 1.185550778167024 gammas[0]
   + 1.03899428175183 gammas[1] + 1.450554690095896 gammas[2]
   - 1.206394289581885 gammas[3] + tau[13,6] = 0
 tau_computation_13_7: - betas[7] - 1.185550778167024 gammas[0]
   + 1.03899428175183 gammas[1] + 1.450554690095896 gammas[2]
   - 1.206394289581885 gammas[3] + tau[13,7] = 0
 Ensure_only_one_R_true_13: R[13,0] + R[13,1] + R[13,2] + R[13,3] + R[13,4]
   + R[13,5] + R[13,6] + R[13,7] = 1
 Define_Q_hat_13: - 2022.801570428241 R[13,0] - 1288.197957597569 R[13,1]
   - 1311.777438179332 R[13,2] - 1338.014207441777 R[13,3]
   - 1327.470453685654 R[13,4] - 1327.470884107489 R[13,5]
   - 1391.80677226795 R[13,6] - 1343.587567901272 R[13,7] + Q_hat[13] = 0
 Adjust_Q_hat_13: Q0_var[13] - Q_hat[13] + Q_hat_adjusted[13] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_13: Total_Demand_Up_to_K_minus_1[13]
   - 217.8812588025846 R[13,0] - 342.2405585580915 R[13,1]
   - 547.4468156719577 R[13,2] - 671.8060534778729 R[13,3]
   - 744.31186875714 R[13,4] - 868.6711064709234 R[13,5]
   - 1020.803320841409 R[13,6] - 1094.869092034218 R[13,7] = 0
 Constr_Total_Demand_from_K_to_T_13: Total_Demand_from_k_to_T[13]
   - 1125.7063086592 R[13,0] - 1001.347008903693 R[13,1]
   - 796.1407517898272 R[13,2] - 671.7815139839121 R[13,3]
   - 599.2756987046449 R[13,4] - 474.9164609908615 R[13,5]
   - 322.7842466203763 R[13,6] - 248.7184754275668 R[13,7] = 0
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
 R294: 1.840234842308595 alphas[0] + 0.8464738653451827 alphas[1]
   - 0.8435804691776236 alphas[2] - 1.548996939710779 alphas[3] - alphas[4]
   + f_var[14] = 0
 Q0_upper_bound_14:
   - 2053.400566507637 Fraction_for_second_order_amount[14] + Q0_var[14]
   = 0
 tau_computation_14_0: - betas[0] + 1.840234842308595 gammas[0]
   + 0.8464738653451827 gammas[1] - 0.8435804691776236 gammas[2]
   - 1.548996939710779 gammas[3] + tau[14,0] = 0
 tau_computation_14_1: - betas[1] + 1.840234842308595 gammas[0]
   + 0.8464738653451827 gammas[1] - 0.8435804691776236 gammas[2]
   - 1.548996939710779 gammas[3] + tau[14,1] = 0
 tau_computation_14_2: - betas[2] + 1.840234842308595 gammas[0]
   + 0.8464738653451827 gammas[1] - 0.8435804691776236 gammas[2]
   - 1.548996939710779 gammas[3] + tau[14,2] = 0
 tau_computation_14_3: - betas[3] + 1.840234842308595 gammas[0]
   + 0.8464738653451827 gammas[1] - 0.8435804691776236 gammas[2]
   - 1.548996939710779 gammas[3] + tau[14,3] = 0
 tau_computation_14_4: - betas[4] + 1.840234842308595 gammas[0]
   + 0.8464738653451827 gammas[1] - 0.8435804691776236 gammas[2]
   - 1.548996939710779 gammas[3] + tau[14,4] = 0
 tau_computation_14_5: - betas[5] + 1.840234842308595 gammas[0]
   + 0.8464738653451827 gammas[1] - 0.8435804691776236 gammas[2]
   - 1.548996939710779 gammas[3] + tau[14,5] = 0
 tau_computation_14_6: - betas[6] + 1.840234842308595 gammas[0]
   + 0.8464738653451827 gammas[1] - 0.8435804691776236 gammas[2]
   - 1.548996939710779 gammas[3] + tau[14,6] = 0
 tau_computation_14_7: - betas[7] + 1.840234842308595 gammas[0]
   + 0.8464738653451827 gammas[1] - 0.8435804691776236 gammas[2]
   - 1.548996939710779 gammas[3] + tau[14,7] = 0
 Ensure_only_one_R_true_14: R[14,0] + R[14,1] + R[14,2] + R[14,3] + R[14,4]
   + R[14,5] + R[14,6] + R[14,7] = 1
 Define_Q_hat_14: - 2585.803813465184 R[14,0] - 2538.074191005085 R[14,1]
   - 2430.780980376641 R[14,2] - 2448.117176646982 R[14,3]
   - 2487.082967858961 R[14,4] - 2487.083560943977 R[14,5]
   - 2388.931753700003 R[14,6] - 2433.972044150916 R[14,7] + Q_hat[14] = 0
 Adjust_Q_hat_14: Q0_var[14] - Q_hat[14] + Q_hat_adjusted[14] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_14: Total_Demand_Up_to_K_minus_1[14]
   - 433.2149138929278 R[14,0] - 686.6643684779915 R[14,1]
   - 972.3380599895331 R[14,2] - 1225.787515445467 R[14,3]
   - 1390.484151114536 R[14,4] - 1643.933606040155 R[14,5]
   - 1757.996879096037 R[14,6] - 1927.073132752433 R[14,7] = 0
 Constr_Total_Demand_from_K_to_T_14: Total_Demand_from_k_to_T[14]
   - 2000.757128710744 R[14,0] - 1747.30767412568 R[14,1]
   - 1461.633982614138 R[14,2] - 1208.184527158205 R[14,3]
   - 1043.487891489136 R[14,4] - 790.0384365635169 R[14,5]
   - 675.975163507635 R[14,6] - 506.8989098512386 R[14,7] = 0
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
 R315: 1.783252692812304 alphas[0] + 0.2789851684846042 alphas[1]
   + 1.010402835029238 alphas[2] - 1.173185428878948 alphas[3] - alphas[4]
   + f_var[15] = 0
 Q0_upper_bound_15:
   - 2053.400566507637 Fraction_for_second_order_amount[15] + Q0_var[15]
   = 0
 tau_computation_15_0: - betas[0] + 1.783252692812304 gammas[0]
   + 0.2789851684846042 gammas[1] + 1.010402835029238 gammas[2]
   - 1.173185428878948 gammas[3] + tau[15,0] = 0
 tau_computation_15_1: - betas[1] + 1.783252692812304 gammas[0]
   + 0.2789851684846042 gammas[1] + 1.010402835029238 gammas[2]
   - 1.173185428878948 gammas[3] + tau[15,1] = 0
 tau_computation_15_2: - betas[2] + 1.783252692812304 gammas[0]
   + 0.2789851684846042 gammas[1] + 1.010402835029238 gammas[2]
   - 1.173185428878948 gammas[3] + tau[15,2] = 0
 tau_computation_15_3: - betas[3] + 1.783252692812304 gammas[0]
   + 0.2789851684846042 gammas[1] + 1.010402835029238 gammas[2]
   - 1.173185428878948 gammas[3] + tau[15,3] = 0
 tau_computation_15_4: - betas[4] + 1.783252692812304 gammas[0]
   + 0.2789851684846042 gammas[1] + 1.010402835029238 gammas[2]
   - 1.173185428878948 gammas[3] + tau[15,4] = 0
 tau_computation_15_5: - betas[5] + 1.783252692812304 gammas[0]
   + 0.2789851684846042 gammas[1] + 1.010402835029238 gammas[2]
   - 1.173185428878948 gammas[3] + tau[15,5] = 0
 tau_computation_15_6: - betas[6] + 1.783252692812304 gammas[0]
   + 0.2789851684846042 gammas[1] + 1.010402835029238 gammas[2]
   - 1.173185428878948 gammas[3] + tau[15,6] = 0
 tau_computation_15_7: - betas[7] + 1.783252692812304 gammas[0]
   + 0.2789851684846042 gammas[1] + 1.010402835029238 gammas[2]
   - 1.173185428878948 gammas[3] + tau[15,7] = 0
 Ensure_only_one_R_true_15: R[15,0] + R[15,1] + R[15,2] + R[15,3] + R[15,4]
   + R[15,5] + R[15,6] + R[15,7] = 1
 Define_Q_hat_15: - 2046.689452969176 R[15,0] - 1521.516242350157 R[15,1]
   - 1401.846082259457 R[15,2] - 1447.305877559308 R[15,3]
   - 1370.860615146246 R[15,4] - 1370.861709557622 R[15,5]
   - 1524.347356989327 R[15,6] - 1504.336492944726 R[15,7] + Q_hat[15] = 0
 Adjust_Q_hat_15: Q0_var[15] - Q_hat[15] + Q_hat_adjusted[15] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_15: Total_Demand_Up_to_K_minus_1[15]
   - 227.0177500395668 R[15,0] - 376.1442655762053 R[15,1]
   - 478.8602860392876 R[15,2] - 627.9867215387166 R[15,3]
   - 699.6665451611801 R[15,4] - 848.7929803729263 R[15,5]
   - 1142.094886741522 R[15,6] - 1206.0836216997 R[15,7] = 0
 Constr_Total_Demand_from_K_to_T_15: Total_Demand_from_k_to_T[15]
   - 1277.318742083625 R[15,0] - 1128.192226546987 R[15,1]
   - 1025.476206083905 R[15,2] - 876.3497705844757 R[15,3]
   - 804.6699469620121 R[15,4] - 655.5435117502659 R[15,5]
   - 362.2416053816698 R[15,6] - 298.2528704234923 R[15,7] = 0
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
 R336: 2.020167489294702 alphas[0] - 1.211968842659354 alphas[1]
   - 1.142610938055166 alphas[2] + 1.476524961653682 alphas[3] - alphas[4]
   + f_var[16] = 0
 Q0_upper_bound_16:
   - 2053.400566507637 Fraction_for_second_order_amount[16] + Q0_var[16]
   = 0
 tau_computation_16_0: - betas[0] + 2.020167489294702 gammas[0]
   - 1.211968842659354 gammas[1] - 1.142610938055166 gammas[2]
   + 1.476524961653682 gammas[3] + tau[16,0] = 0
 tau_computation_16_1: - betas[1] + 2.020167489294702 gammas[0]
   - 1.211968842659354 gammas[1] - 1.142610938055166 gammas[2]
   + 1.476524961653682 gammas[3] + tau[16,1] = 0
 tau_computation_16_2: - betas[2] + 2.020167489294702 gammas[0]
   - 1.211968842659354 gammas[1] - 1.142610938055166 gammas[2]
   + 1.476524961653682 gammas[3] + tau[16,2] = 0
 tau_computation_16_3: - betas[3] + 2.020167489294702 gammas[0]
   - 1.211968842659354 gammas[1] - 1.142610938055166 gammas[2]
   + 1.476524961653682 gammas[3] + tau[16,3] = 0
 tau_computation_16_4: - betas[4] + 2.020167489294702 gammas[0]
   - 1.211968842659354 gammas[1] - 1.142610938055166 gammas[2]
   + 1.476524961653682 gammas[3] + tau[16,4] = 0
 tau_computation_16_5: - betas[5] + 2.020167489294702 gammas[0]
   - 1.211968842659354 gammas[1] - 1.142610938055166 gammas[2]
   + 1.476524961653682 gammas[3] + tau[16,5] = 0
 tau_computation_16_6: - betas[6] + 2.020167489294702 gammas[0]
   - 1.211968842659354 gammas[1] - 1.142610938055166 gammas[2]
   + 1.476524961653682 gammas[3] + tau[16,6] = 0
 tau_computation_16_7: - betas[7] + 2.020167489294702 gammas[0]
   - 1.211968842659354 gammas[1] - 1.142610938055166 gammas[2]
   + 1.476524961653682 gammas[3] + tau[16,7] = 0
 Ensure_only_one_R_true_16: R[16,0] + R[16,1] + R[16,2] + R[16,3] + R[16,4]
   + R[16,5] + R[16,6] + R[16,7] = 1
 Define_Q_hat_16: - 2185.774795555209 R[16,0] - 2664.510384723219 R[16,1]
   - 2658.794508568069 R[16,2] - 2651.248828352956 R[16,3]
   - 2588.733522947857 R[16,4] - 2588.732982316455 R[16,5]
   - 2644.409220069302 R[16,6] - 2446.578434794819 R[16,7] + Q_hat[16] = 0
 Adjust_Q_hat_16: Q0_var[16] - Q_hat[16] + Q_hat_adjusted[16] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_16: Total_Demand_Up_to_K_minus_1[16]
   - 280.2142613482757 R[16,0] - 550.4900602111945 R[16,1]
   - 815.3833265438661 R[16,2] - 1085.659125541144 R[16,3]
   - 1273.291250356725 R[16,4] - 1543.567049634515 R[16,5]
   - 1859.572000568232 R[16,6] - 1906.026838820684 R[16,7] = 0
 Constr_Total_Demand_from_K_to_T_16: Total_Demand_from_k_to_T[16]
   - 2166.36417602799 R[16,0] - 1896.088377165071 R[16,1]
   - 1631.195110832399 R[16,2] - 1360.919311835121 R[16,3]
   - 1173.28718701954 R[16,4] - 903.0113877417498 R[16,5]
   - 587.0064368080334 R[16,6] - 540.5515985555808 R[16,7] = 0
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
 R357: - 0.8563431724362236 alphas[0] + 1.174424780788562 alphas[1]
   + 0.1895025714961081 alphas[2] + 0.3832772058522401 alphas[3]
   - alphas[4] + f_var[17] = 0
 Q0_upper_bound_17:
   - 2053.400566507637 Fraction_for_second_order_amount[17] + Q0_var[17]
   = 0
 tau_computation_17_0: - betas[0] - 0.8563431724362236 gammas[0]
   + 1.174424780788562 gammas[1] + 0.1895025714961081 gammas[2]
   + 0.3832772058522401 gammas[3] + tau[17,0] = 0
 tau_computation_17_1: - betas[1] - 0.8563431724362236 gammas[0]
   + 1.174424780788562 gammas[1] + 0.1895025714961081 gammas[2]
   + 0.3832772058522401 gammas[3] + tau[17,1] = 0
 tau_computation_17_2: - betas[2] - 0.8563431724362236 gammas[0]
   + 1.174424780788562 gammas[1] + 0.1895025714961081 gammas[2]
   + 0.3832772058522401 gammas[3] + tau[17,2] = 0
 tau_computation_17_3: - betas[3] - 0.8563431724362236 gammas[0]
   + 1.174424780788562 gammas[1] + 0.1895025714961081 gammas[2]
   + 0.3832772058522401 gammas[3] + tau[17,3] = 0
 tau_computation_17_4: - betas[4] - 0.8563431724362236 gammas[0]
   + 1.174424780788562 gammas[1] + 0.1895025714961081 gammas[2]
   + 0.3832772058522401 gammas[3] + tau[17,4] = 0
 tau_computation_17_5: - betas[5] - 0.8563431724362236 gammas[0]
   + 1.174424780788562 gammas[1] + 0.1895025714961081 gammas[2]
   + 0.3832772058522401 gammas[3] + tau[17,5] = 0
 tau_computation_17_6: - betas[6] - 0.8563431724362236 gammas[0]
   + 1.174424780788562 gammas[1] + 0.1895025714961081 gammas[2]
   + 0.3832772058522401 gammas[3] + tau[17,6] = 0
 tau_computation_17_7: - betas[7] - 0.8563431724362236 gammas[0]
   + 1.174424780788562 gammas[1] + 0.1895025714961081 gammas[2]
   + 0.3832772058522401 gammas[3] + tau[17,7] = 0
 Ensure_only_one_R_true_17: R[17,0] + R[17,1] + R[17,2] + R[17,3] + R[17,4]
   + R[17,5] + R[17,6] + R[17,7] = 1
 Define_Q_hat_17: - 1593.691325408947 R[17,0] - 1918.323112283352 R[17,1]
   - 1950.805385529675 R[17,2] - 1953.146214606116 R[17,3]
   - 1848.610994499188 R[17,4] - 1848.611061216892 R[17,5]
   - 1842.956583651056 R[17,6] - 1786.645402830598 R[17,7] + Q_hat[17] = 0
 Adjust_Q_hat_17: Q0_var[17] - Q_hat[17] + Q_hat_adjusted[17] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_17: Total_Demand_Up_to_K_minus_1[17]
   - 53.75779634836164 R[17,0] - 249.076005973909 R[17,1]
   - 365.5903633577287 R[17,2] - 560.90858130426 R[17,3]
   - 765.1782978188107 R[17,4] - 960.4965159777754 R[17,5]
   - 1220.982504970496 R[17,6] - 1396.008966051397 R[17,7] = 0
 Constr_Total_Demand_from_K_to_T_17: Total_Demand_from_k_to_T[17]
   - 1732.887606020964 R[17,0] - 1537.569396395417 R[17,1]
   - 1421.055039011597 R[17,2] - 1225.736821065066 R[17,3]
   - 1021.467104550515 R[17,4] - 826.1488863915504 R[17,5]
   - 565.66289739883 R[17,6] - 390.6364363179293 R[17,7] = 0
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
 R378: - 0.6635029545277169 alphas[0] - 1.267852010342292 alphas[1]
   - 1.145419577230695 alphas[2] - 0.7782420895307338 alphas[3] - alphas[4]
   + f_var[18] = 0
 Q0_upper_bound_18:
   - 2053.400566507637 Fraction_for_second_order_amount[18] + Q0_var[18]
   = 0
 tau_computation_18_0: - betas[0] - 0.6635029545277169 gammas[0]
   - 1.267852010342292 gammas[1] - 1.145419577230695 gammas[2]
   - 0.7782420895307338 gammas[3] + tau[18,0] = 0
 tau_computation_18_1: - betas[1] - 0.6635029545277169 gammas[0]
   - 1.267852010342292 gammas[1] - 1.145419577230695 gammas[2]
   - 0.7782420895307338 gammas[3] + tau[18,1] = 0
 tau_computation_18_2: - betas[2] - 0.6635029545277169 gammas[0]
   - 1.267852010342292 gammas[1] - 1.145419577230695 gammas[2]
   - 0.7782420895307338 gammas[3] + tau[18,2] = 0
 tau_computation_18_3: - betas[3] - 0.6635029545277169 gammas[0]
   - 1.267852010342292 gammas[1] - 1.145419577230695 gammas[2]
   - 0.7782420895307338 gammas[3] + tau[18,3] = 0
 tau_computation_18_4: - betas[4] - 0.6635029545277169 gammas[0]
   - 1.267852010342292 gammas[1] - 1.145419577230695 gammas[2]
   - 0.7782420895307338 gammas[3] + tau[18,4] = 0
 tau_computation_18_5: - betas[5] - 0.6635029545277169 gammas[0]
   - 1.267852010342292 gammas[1] - 1.145419577230695 gammas[2]
   - 0.7782420895307338 gammas[3] + tau[18,5] = 0
 tau_computation_18_6: - betas[6] - 0.6635029545277169 gammas[0]
   - 1.267852010342292 gammas[1] - 1.145419577230695 gammas[2]
   - 0.7782420895307338 gammas[3] + tau[18,6] = 0
 tau_computation_18_7: - betas[7] - 0.6635029545277169 gammas[0]
   - 1.267852010342292 gammas[1] - 1.145419577230695 gammas[2]
   - 0.7782420895307338 gammas[3] + tau[18,7] = 0
 Ensure_only_one_R_true_18: R[18,0] + R[18,1] + R[18,2] + R[18,3] + R[18,4]
   + R[18,5] + R[18,6] + R[18,7] = 1
 Define_Q_hat_18: - 2154.799619889661 R[18,0] - 2663.600618919495 R[18,1]
   - 2732.965137661076 R[18,2] - 2711.823838063111 R[18,3]
   - 2607.071895438658 R[18,4] - 2607.07222065821 R[18,5]
   - 2598.793968600316 R[18,6] - 2791.371400639382 R[18,7] + Q_hat[18] = 0
 Adjust_Q_hat_18: Q0_var[18] - Q_hat[18] + Q_hat_adjusted[18] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_18: Total_Demand_Up_to_K_minus_1[18]
   - 268.3670655766412 R[18,0] - 538.8009066876389 R[18,1]
   - 856.0260335889975 R[18,2] - 1126.459874436234 R[18,3]
   - 1269.199096099336 R[18,4] - 1539.632936563049 R[18,5]
   - 1785.771852633308 R[18,6] - 2250.503717816093 R[18,7] = 0
 Constr_Total_Demand_from_K_to_T_18: Total_Demand_from_k_to_T[18]
   - 2523.004333166878 R[18,0] - 2252.57049205588 R[18,1]
   - 1935.345365154521 R[18,2] - 1664.911524307286 R[18,3]
   - 1522.172302644183 R[18,4] - 1251.73846218047 R[18,5]
   - 1005.599546110211 R[18,6] - 540.8676809274264 R[18,7] = 0
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
 R399: - 0.9887399790034055 alphas[0] + 1.177730850908782 alphas[1]
   - 0.3048078049445529 alphas[2] + 0.9745282776980045 alphas[3]
   - alphas[4] + f_var[19] = 0
 Q0_upper_bound_19:
   - 2053.400566507637 Fraction_for_second_order_amount[19] + Q0_var[19]
   = 0
 tau_computation_19_0: - betas[0] - 0.9887399790034055 gammas[0]
   + 1.177730850908782 gammas[1] - 0.3048078049445529 gammas[2]
   + 0.9745282776980045 gammas[3] + tau[19,0] = 0
 tau_computation_19_1: - betas[1] - 0.9887399790034055 gammas[0]
   + 1.177730850908782 gammas[1] - 0.3048078049445529 gammas[2]
   + 0.9745282776980045 gammas[3] + tau[19,1] = 0
 tau_computation_19_2: - betas[2] - 0.9887399790034055 gammas[0]
   + 1.177730850908782 gammas[1] - 0.3048078049445529 gammas[2]
   + 0.9745282776980045 gammas[3] + tau[19,2] = 0
 tau_computation_19_3: - betas[3] - 0.9887399790034055 gammas[0]
   + 1.177730850908782 gammas[1] - 0.3048078049445529 gammas[2]
   + 0.9745282776980045 gammas[3] + tau[19,3] = 0
 tau_computation_19_4: - betas[4] - 0.9887399790034055 gammas[0]
   + 1.177730850908782 gammas[1] - 0.3048078049445529 gammas[2]
   + 0.9745282776980045 gammas[3] + tau[19,4] = 0
 tau_computation_19_5: - betas[5] - 0.9887399790034055 gammas[0]
   + 1.177730850908782 gammas[1] - 0.3048078049445529 gammas[2]
   + 0.9745282776980045 gammas[3] + tau[19,5] = 0
 tau_computation_19_6: - betas[6] - 0.9887399790034055 gammas[0]
   + 1.177730850908782 gammas[1] - 0.3048078049445529 gammas[2]
   + 0.9745282776980045 gammas[3] + tau[19,6] = 0
 tau_computation_19_7: - betas[7] - 0.9887399790034055 gammas[0]
   + 1.177730850908782 gammas[1] - 0.3048078049445529 gammas[2]
   + 0.9745282776980045 gammas[3] + tau[19,7] = 0
 Ensure_only_one_R_true_19: R[19,0] + R[19,1] + R[19,2] + R[19,3] + R[19,4]
   + R[19,5] + R[19,6] + R[19,7] = 1
 Define_Q_hat_19: - 2284.983827512279 R[19,0] - 2231.551558546518 R[19,1]
   - 2242.158861124781 R[19,2] - 2244.949411602609 R[19,3]
   - 2135.38682900926 R[19,4] - 2135.387089657183 R[19,5]
   - 2129.281185047815 R[19,6] - 2149.588554172074 R[19,7] + Q_hat[19] = 0
 Adjust_Q_hat_19: Q0_var[19] - Q_hat[19] + Q_hat_adjusted[19] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_19: Total_Demand_Up_to_K_minus_1[19]
   - 318.1591252160536 R[19,0] - 541.2920242530778 R[19,1]
   - 833.1843297576704 R[19,2] - 1056.317227447236 R[19,3]
   - 1105.42289261011 R[19,4] - 1328.555790129551 R[19,5]
   - 1504.055676300661 R[19,6] - 1703.322760250565 R[19,7] = 0
 Constr_Total_Demand_from_K_to_T_19: Total_Demand_from_k_to_T[19]
   - 1831.429430073394 R[19,0] - 1608.29653103637 R[19,1]
   - 1316.404225531777 R[19,2] - 1093.271327842211 R[19,3]
   - 1044.165662679338 R[19,4] - 821.0327651598968 R[19,5]
   - 645.5328789887867 R[19,6] - 446.2657950388829 R[19,7] = 0
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
 R420: - 1.373287391687492 alphas[0] - 1.724481283476159 alphas[1]
   - 0.450191155178684 alphas[2] - 0.1223073485596939 alphas[3] - alphas[4]
   + f_var[20] = 0
 Q0_upper_bound_20:
   - 2053.400566507637 Fraction_for_second_order_amount[20] + Q0_var[20]
   = 0
 tau_computation_20_0: - betas[0] - 1.373287391687492 gammas[0]
   - 1.724481283476159 gammas[1] - 0.450191155178684 gammas[2]
   - 0.1223073485596939 gammas[3] + tau[20,0] = 0
 tau_computation_20_1: - betas[1] - 1.373287391687492 gammas[0]
   - 1.724481283476159 gammas[1] - 0.450191155178684 gammas[2]
   - 0.1223073485596939 gammas[3] + tau[20,1] = 0
 tau_computation_20_2: - betas[2] - 1.373287391687492 gammas[0]
   - 1.724481283476159 gammas[1] - 0.450191155178684 gammas[2]
   - 0.1223073485596939 gammas[3] + tau[20,2] = 0
 tau_computation_20_3: - betas[3] - 1.373287391687492 gammas[0]
   - 1.724481283476159 gammas[1] - 0.450191155178684 gammas[2]
   - 0.1223073485596939 gammas[3] + tau[20,3] = 0
 tau_computation_20_4: - betas[4] - 1.373287391687492 gammas[0]
   - 1.724481283476159 gammas[1] - 0.450191155178684 gammas[2]
   - 0.1223073485596939 gammas[3] + tau[20,4] = 0
 tau_computation_20_5: - betas[5] - 1.373287391687492 gammas[0]
   - 1.724481283476159 gammas[1] - 0.450191155178684 gammas[2]
   - 0.1223073485596939 gammas[3] + tau[20,5] = 0
 tau_computation_20_6: - betas[6] - 1.373287391687492 gammas[0]
   - 1.724481283476159 gammas[1] - 0.450191155178684 gammas[2]
   - 0.1223073485596939 gammas[3] + tau[20,6] = 0
 tau_computation_20_7: - betas[7] - 1.373287391687492 gammas[0]
   - 1.724481283476159 gammas[1] - 0.450191155178684 gammas[2]
   - 0.1223073485596939 gammas[3] + tau[20,7] = 0
 Ensure_only_one_R_true_20: R[20,0] + R[20,1] + R[20,2] + R[20,3] + R[20,4]
   + R[20,5] + R[20,6] + R[20,7] = 1
 Define_Q_hat_20: - 2021.039104822414 R[20,0] - 2287.670473791569 R[20,1]
   - 2311.087993584502 R[20,2] - 2307.967103723448 R[20,3]
   - 2306.799348486382 R[20,4] - 2306.79925490274 R[20,5]
   - 2160.754646309271 R[20,6] - 2130.567076467601 R[20,7] + Q_hat[20] = 0
 Adjust_Q_hat_20: Q0_var[20] - Q_hat[20] + Q_hat_adjusted[20] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_20: Total_Demand_Up_to_K_minus_1[20]
   - 217.2071617353573 R[20,0] - 448.5207385437724 R[20,1]
   - 682.1524685525344 R[20,2] - 913.4660385107381 R[20,3]
   - 1142.22597777693 R[20,4] - 1373.53954773184 R[20,5]
   - 1486.118567034667 R[20,6] - 1667.939937847112 R[20,7] = 0
 Constr_Total_Demand_from_K_to_T_20: Total_Demand_from_k_to_T[20]
   - 1913.359916021575 R[20,0] - 1682.04633921316 R[20,1]
   - 1448.414609204398 R[20,2] - 1217.101039246194 R[20,3]
   - 988.3410999800026 R[20,4] - 757.0275300250925 R[20,5]
   - 644.4485107222661 R[20,6] - 462.6271399098202 R[20,7] = 0
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
 R441: - 0.7378651309105446 alphas[0] - 0.0504325599864221 alphas[1]
   + 0.5447050171735942 alphas[2] + 1.342722759310279 alphas[3] - alphas[4]
   + f_var[21] = 0
 Q0_upper_bound_21:
   - 2053.400566507637 Fraction_for_second_order_amount[21] + Q0_var[21]
   = 0
 tau_computation_21_0: - betas[0] - 0.7378651309105446 gammas[0]
   - 0.0504325599864221 gammas[1] + 0.5447050171735942 gammas[2]
   + 1.342722759310279 gammas[3] + tau[21,0] = 0
 tau_computation_21_1: - betas[1] - 0.7378651309105446 gammas[0]
   - 0.0504325599864221 gammas[1] + 0.5447050171735942 gammas[2]
   + 1.342722759310279 gammas[3] + tau[21,1] = 0
 tau_computation_21_2: - betas[2] - 0.7378651309105446 gammas[0]
   - 0.0504325599864221 gammas[1] + 0.5447050171735942 gammas[2]
   + 1.342722759310279 gammas[3] + tau[21,2] = 0
 tau_computation_21_3: - betas[3] - 0.7378651309105446 gammas[0]
   - 0.0504325599864221 gammas[1] + 0.5447050171735942 gammas[2]
   + 1.342722759310279 gammas[3] + tau[21,3] = 0
 tau_computation_21_4: - betas[4] - 0.7378651309105446 gammas[0]
   - 0.0504325599864221 gammas[1] + 0.5447050171735942 gammas[2]
   + 1.342722759310279 gammas[3] + tau[21,4] = 0
 tau_computation_21_5: - betas[5] - 0.7378651309105446 gammas[0]
   - 0.0504325599864221 gammas[1] + 0.5447050171735942 gammas[2]
   + 1.342722759310279 gammas[3] + tau[21,5] = 0
 tau_computation_21_6: - betas[6] - 0.7378651309105446 gammas[0]
   - 0.0504325599864221 gammas[1] + 0.5447050171735942 gammas[2]
   + 1.342722759310279 gammas[3] + tau[21,6] = 0
 tau_computation_21_7: - betas[7] - 0.7378651309105446 gammas[0]
   - 0.0504325599864221 gammas[1] + 0.5447050171735942 gammas[2]
   + 1.342722759310279 gammas[3] + tau[21,7] = 0
 Ensure_only_one_R_true_21: R[21,0] + R[21,1] + R[21,2] + R[21,3] + R[21,4]
   + R[21,5] + R[21,6] + R[21,7] = 1
 Define_Q_hat_21: - 1567.812076471181 R[21,0] - 1729.526555973846 R[21,1]
   - 1754.360466408426 R[21,2] - 1763.058471742364 R[21,3]
   - 1670.366119588005 R[21,4] - 1670.366580682236 R[21,5]
   - 1660.948976044534 R[21,6] - 1605.076793335005 R[21,7] + Q_hat[21] = 0
 Adjust_Q_hat_21: Q0_var[21] - Q_hat[21] + Q_hat_adjusted[21] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_21: Total_Demand_Up_to_K_minus_1[21]
   - 43.85965946936432 R[21,0] - 219.1907267827047 R[21,1]
   - 317.1631755567751 R[21,2] - 492.4942710461773 R[21,3]
   - 688.4827988465235 R[21,4] - 863.813894328199 R[21,5]
   - 1101.208289110015 R[21,6] - 1254.414601510171 R[21,7] = 0
 Constr_Total_Demand_from_K_to_T_21: Total_Demand_from_k_to_T[21]
   - 1561.217133004158 R[21,0] - 1385.886065690817 R[21,1]
   - 1287.913616916747 R[21,2] - 1112.582521427345 R[21,3]
   - 916.5939936269987 R[21,4] - 741.2628981453231 R[21,5]
   - 503.8685033635074 R[21,6] - 350.6621909633511 R[21,7] = 0
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
 R462: 0.4577726008491022 alphas[0] - 1.725475619194501 alphas[1]
   - 1.475740864355606 alphas[2] + 0.8854809048116157 alphas[3] - alphas[4]
   + f_var[22] = 0
 Q0_upper_bound_22:
   - 2053.400566507637 Fraction_for_second_order_amount[22] + Q0_var[22]
   = 0
 tau_computation_22_0: - betas[0] + 0.4577726008491022 gammas[0]
   - 1.725475619194501 gammas[1] - 1.475740864355606 gammas[2]
   + 0.8854809048116157 gammas[3] + tau[22,0] = 0
 tau_computation_22_1: - betas[1] + 0.4577726008491022 gammas[0]
   - 1.725475619194501 gammas[1] - 1.475740864355606 gammas[2]
   + 0.8854809048116157 gammas[3] + tau[22,1] = 0
 tau_computation_22_2: - betas[2] + 0.4577726008491022 gammas[0]
   - 1.725475619194501 gammas[1] - 1.475740864355606 gammas[2]
   + 0.8854809048116157 gammas[3] + tau[22,2] = 0
 tau_computation_22_3: - betas[3] + 0.4577726008491022 gammas[0]
   - 1.725475619194501 gammas[1] - 1.475740864355606 gammas[2]
   + 0.8854809048116157 gammas[3] + tau[22,3] = 0
 tau_computation_22_4: - betas[4] + 0.4577726008491022 gammas[0]
   - 1.725475619194501 gammas[1] - 1.475740864355606 gammas[2]
   + 0.8854809048116157 gammas[3] + tau[22,4] = 0
 tau_computation_22_5: - betas[5] + 0.4577726008491022 gammas[0]
   - 1.725475619194501 gammas[1] - 1.475740864355606 gammas[2]
   + 0.8854809048116157 gammas[3] + tau[22,5] = 0
 tau_computation_22_6: - betas[6] + 0.4577726008491022 gammas[0]
   - 1.725475619194501 gammas[1] - 1.475740864355606 gammas[2]
   + 0.8854809048116157 gammas[3] + tau[22,6] = 0
 tau_computation_22_7: - betas[7] + 0.4577726008491022 gammas[0]
   - 1.725475619194501 gammas[1] - 1.475740864355606 gammas[2]
   + 0.8854809048116157 gammas[3] + tau[22,7] = 0
 Ensure_only_one_R_true_22: R[22,0] + R[22,1] + R[22,2] + R[22,3] + R[22,4]
   + R[22,5] + R[22,6] + R[22,7] = 1
 Define_Q_hat_22: - 2355.749684765982 R[22,0] - 2852.802301077008 R[22,1]
   - 3011.722845958367 R[22,2] - 2970.759993808791 R[22,3]
   - 3061.624510332111 R[22,4] - 3061.623574916186 R[22,5]
   - 3099.82087170999 R[22,6] - 3034.176040048083 R[22,7] + Q_hat[22] = 0
 Adjust_Q_hat_22: Q0_var[22] - Q_hat[22] + Q_hat_adjusted[22] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_22: Total_Demand_Up_to_K_minus_1[22]
   - 345.2252175567221 R[22,0] - 634.2461263558405 R[22,1]
   - 1082.718046781643 R[22,2] - 1371.738955484279 R[22,3]
   - 1639.852421770204 R[22,4] - 1928.87333045101 R[22,5]
   - 2185.476970530086 R[22,6] - 2456.134222796946 R[22,7] = 0
 Constr_Total_Demand_from_K_to_T_22: Total_Demand_from_k_to_T[22]
   - 2688.950822601836 R[22,0] - 2399.929913802717 R[22,1]
   - 1951.457993376915 R[22,2] - 1662.437084674279 R[22,3]
   - 1394.323618388354 R[22,4] - 1105.302709707548 R[22,5]
   - 848.699069628472 R[22,6] - 578.0418173616119 R[22,7] = 0
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
 R483: - 0.6719031202226472 alphas[0] - 0.4994128952930674 alphas[1]
   - 0.0891436867992024 alphas[2] + 1.455154637985153 alphas[3] - alphas[4]
   + f_var[23] = 0
 Q0_upper_bound_23:
   - 2053.400566507637 Fraction_for_second_order_amount[23] + Q0_var[23]
   = 0
 tau_computation_23_0: - betas[0] - 0.6719031202226472 gammas[0]
   - 0.4994128952930674 gammas[1] - 0.0891436867992024 gammas[2]
   + 1.455154637985153 gammas[3] + tau[23,0] = 0
 tau_computation_23_1: - betas[1] - 0.6719031202226472 gammas[0]
   - 0.4994128952930674 gammas[1] - 0.0891436867992024 gammas[2]
   + 1.455154637985153 gammas[3] + tau[23,1] = 0
 tau_computation_23_2: - betas[2] - 0.6719031202226472 gammas[0]
   - 0.4994128952930674 gammas[1] - 0.0891436867992024 gammas[2]
   + 1.455154637985153 gammas[3] + tau[23,2] = 0
 tau_computation_23_3: - betas[3] - 0.6719031202226472 gammas[0]
   - 0.4994128952930674 gammas[1] - 0.0891436867992024 gammas[2]
   + 1.455154637985153 gammas[3] + tau[23,3] = 0
 tau_computation_23_4: - betas[4] - 0.6719031202226472 gammas[0]
   - 0.4994128952930674 gammas[1] - 0.0891436867992024 gammas[2]
   + 1.455154637985153 gammas[3] + tau[23,4] = 0
 tau_computation_23_5: - betas[5] - 0.6719031202226472 gammas[0]
   - 0.4994128952930674 gammas[1] - 0.0891436867992024 gammas[2]
   + 1.455154637985153 gammas[3] + tau[23,5] = 0
 tau_computation_23_6: - betas[6] - 0.6719031202226472 gammas[0]
   - 0.4994128952930674 gammas[1] - 0.0891436867992024 gammas[2]
   + 1.455154637985153 gammas[3] + tau[23,6] = 0
 tau_computation_23_7: - betas[7] - 0.6719031202226472 gammas[0]
   - 0.4994128952930674 gammas[1] - 0.0891436867992024 gammas[2]
   + 1.455154637985153 gammas[3] + tau[23,7] = 0
 Ensure_only_one_R_true_23: R[23,0] + R[23,1] + R[23,2] + R[23,3] + R[23,4]
   + R[23,5] + R[23,6] + R[23,7] = 1
 Define_Q_hat_23: - 2261.956769485609 R[23,0] - 2116.358722764682 R[23,1]
   - 2276.232288178615 R[23,2] - 2255.624800161222 R[23,3]
   - 2311.076867851932 R[23,4] - 2311.076985669481 R[23,5]
   - 2316.618421733958 R[23,6] - 2259.4796675568 R[23,7] + Q_hat[23] = 0
 Adjust_Q_hat_23: Q0_var[23] - Q_hat[23] + Q_hat_adjusted[23] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_23: Total_Demand_Up_to_K_minus_1[23]
   - 309.3518768752421 R[23,0] - 520.3494351670573 R[23,1]
   - 923.2476748639084 R[23,2] - 1134.245225158536 R[23,3]
   - 1289.022264987886 R[23,4] - 1500.019814795296 R[23,5]
   - 1639.869291127235 R[23,6] - 1837.484567604658 R[23,7] = 0
 Constr_Total_Demand_from_K_to_T_23: Total_Demand_from_k_to_T[23]
   - 1950.127790344235 R[23,0] - 1739.13023205242 R[23,1]
   - 1336.231992355569 R[23,2] - 1125.234442060941 R[23,3]
   - 970.457402231591 R[23,4] - 759.4598524241816 R[23,5]
   - 619.6103760922421 R[23,6] - 421.9950996148187 R[23,7] = 0
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
 R504: 1.67297557835104 alphas[0] - 0.9425375890211247 alphas[1]
   + 0.888892836523537 alphas[2] - 0.9782274247791758 alphas[3] - alphas[4]
   + f_var[24] = 0
 Q0_upper_bound_24:
   - 2053.400566507637 Fraction_for_second_order_amount[24] + Q0_var[24]
   = 0
 tau_computation_24_0: - betas[0] + 1.67297557835104 gammas[0]
   - 0.9425375890211247 gammas[1] + 0.888892836523537 gammas[2]
   - 0.9782274247791758 gammas[3] + tau[24,0] = 0
 tau_computation_24_1: - betas[1] + 1.67297557835104 gammas[0]
   - 0.9425375890211247 gammas[1] + 0.888892836523537 gammas[2]
   - 0.9782274247791758 gammas[3] + tau[24,1] = 0
 tau_computation_24_2: - betas[2] + 1.67297557835104 gammas[0]
   - 0.9425375890211247 gammas[1] + 0.888892836523537 gammas[2]
   - 0.9782274247791758 gammas[3] + tau[24,2] = 0
 tau_computation_24_3: - betas[3] + 1.67297557835104 gammas[0]
   - 0.9425375890211247 gammas[1] + 0.888892836523537 gammas[2]
   - 0.9782274247791758 gammas[3] + tau[24,3] = 0
 tau_computation_24_4: - betas[4] + 1.67297557835104 gammas[0]
   - 0.9425375890211247 gammas[1] + 0.888892836523537 gammas[2]
   - 0.9782274247791758 gammas[3] + tau[24,4] = 0
 tau_computation_24_5: - betas[5] + 1.67297557835104 gammas[0]
   - 0.9425375890211247 gammas[1] + 0.888892836523537 gammas[2]
   - 0.9782274247791758 gammas[3] + tau[24,5] = 0
 tau_computation_24_6: - betas[6] + 1.67297557835104 gammas[0]
   - 0.9425375890211247 gammas[1] + 0.888892836523537 gammas[2]
   - 0.9782274247791758 gammas[3] + tau[24,6] = 0
 tau_computation_24_7: - betas[7] + 1.67297557835104 gammas[0]
   - 0.9425375890211247 gammas[1] + 0.888892836523537 gammas[2]
   - 0.9782274247791758 gammas[3] + tau[24,7] = 0
 Ensure_only_one_R_true_24: R[24,0] + R[24,1] + R[24,2] + R[24,3] + R[24,4]
   + R[24,5] + R[24,6] + R[24,7] = 1
 Define_Q_hat_24: - 1937.671827061494 R[24,0] - 1577.02366911822 R[24,1]
   - 1653.343827993618 R[24,2] - 1668.172281737045 R[24,3]
   - 1622.071286195245 R[24,4] - 1622.071867605765 R[24,5]
   - 1592.986828844256 R[24,6] - 1592.163090652771 R[24,7] + Q_hat[24] = 0
 Adjust_Q_hat_24: Q0_var[24] - Q_hat[24] + Q_hat_adjusted[24] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_24: Total_Demand_Up_to_K_minus_1[24]
   - 185.3213551382904 R[24,0] - 341.2857806370553 R[24,1]
   - 574.7974229432102 R[24,2] - 730.761185311175 R[24,3]
   - 832.4181867481882 R[24,4] - 988.3819488877331 R[24,5]
   - 1104.186932657501 R[24,6] - 1280.235565652873 R[24,7] = 0
 Constr_Total_Demand_from_K_to_T_24: Total_Demand_from_k_to_T[24]
   - 1406.841734793673 R[24,0] - 1250.877309294908 R[24,1]
   - 1017.365666988753 R[24,2] - 861.4019046207882 R[24,3]
   - 759.744903183775 R[24,4] - 603.7811410442301 R[24,5]
   - 487.9761572744624 R[24,6] - 311.9275242790899 R[24,7] = 0
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
 softmax_0_0: - exp_tau_var_0_0 + [ r[0,0] * exp_tau_var_0_0
   + r[0,0] * exp_tau_var_0_1 + r[0,0] * exp_tau_var_0_2
   + r[0,0] * exp_tau_var_0_3 + r[0,0] * exp_tau_var_0_4
   + r[0,0] * exp_tau_var_0_5 + r[0,0] * exp_tau_var_0_6
   + r[0,0] * exp_tau_var_0_7 ] = 0
 softmax_0_1: - exp_tau_var_0_1 + [ r[0,1] * exp_tau_var_0_0
   + r[0,1] * exp_tau_var_0_1 + r[0,1] * exp_tau_var_0_2
   + r[0,1] * exp_tau_var_0_3 + r[0,1] * exp_tau_var_0_4
   + r[0,1] * exp_tau_var_0_5 + r[0,1] * exp_tau_var_0_6
   + r[0,1] * exp_tau_var_0_7 ] = 0
 softmax_0_2: - exp_tau_var_0_2 + [ r[0,2] * exp_tau_var_0_0
   + r[0,2] * exp_tau_var_0_1 + r[0,2] * exp_tau_var_0_2
   + r[0,2] * exp_tau_var_0_3 + r[0,2] * exp_tau_var_0_4
   + r[0,2] * exp_tau_var_0_5 + r[0,2] * exp_tau_var_0_6
   + r[0,2] * exp_tau_var_0_7 ] = 0
 softmax_0_3: - exp_tau_var_0_3 + [ r[0,3] * exp_tau_var_0_0
   + r[0,3] * exp_tau_var_0_1 + r[0,3] * exp_tau_var_0_2
   + r[0,3] * exp_tau_var_0_3 + r[0,3] * exp_tau_var_0_4
   + r[0,3] * exp_tau_var_0_5 + r[0,3] * exp_tau_var_0_6
   + r[0,3] * exp_tau_var_0_7 ] = 0
 softmax_0_4: - exp_tau_var_0_4 + [ r[0,4] * exp_tau_var_0_0
   + r[0,4] * exp_tau_var_0_1 + r[0,4] * exp_tau_var_0_2
   + r[0,4] * exp_tau_var_0_3 + r[0,4] * exp_tau_var_0_4
   + r[0,4] * exp_tau_var_0_5 + r[0,4] * exp_tau_var_0_6
   + r[0,4] * exp_tau_var_0_7 ] = 0
 softmax_0_5: - exp_tau_var_0_5 + [ r[0,5] * exp_tau_var_0_0
   + r[0,5] * exp_tau_var_0_1 + r[0,5] * exp_tau_var_0_2
   + r[0,5] * exp_tau_var_0_3 + r[0,5] * exp_tau_var_0_4
   + r[0,5] * exp_tau_var_0_5 + r[0,5] * exp_tau_var_0_6
   + r[0,5] * exp_tau_var_0_7 ] = 0
 softmax_0_6: - exp_tau_var_0_6 + [ r[0,6] * exp_tau_var_0_0
   + r[0,6] * exp_tau_var_0_1 + r[0,6] * exp_tau_var_0_2
   + r[0,6] * exp_tau_var_0_3 + r[0,6] * exp_tau_var_0_4
   + r[0,6] * exp_tau_var_0_5 + r[0,6] * exp_tau_var_0_6
   + r[0,6] * exp_tau_var_0_7 ] = 0
 softmax_0_7: - exp_tau_var_0_7 + [ r[0,7] * exp_tau_var_0_0
   + r[0,7] * exp_tau_var_0_1 + r[0,7] * exp_tau_var_0_2
   + r[0,7] * exp_tau_var_0_3 + r[0,7] * exp_tau_var_0_4
   + r[0,7] * exp_tau_var_0_5 + r[0,7] * exp_tau_var_0_6
   + r[0,7] * exp_tau_var_0_7 ] = 0
 softmax_1_0: - exp_tau_var_1_0 + [ r[1,0] * exp_tau_var_1_0
   + r[1,0] * exp_tau_var_1_1 + r[1,0] * exp_tau_var_1_2
   + r[1,0] * exp_tau_var_1_3 + r[1,0] * exp_tau_var_1_4
   + r[1,0] * exp_tau_var_1_5 + r[1,0] * exp_tau_var_1_6
   + r[1,0] * exp_tau_var_1_7 ] = 0
 softmax_1_1: - exp_tau_var_1_1 + [ r[1,1] * exp_tau_var_1_0
   + r[1,1] * exp_tau_var_1_1 + r[1,1] * exp_tau_var_1_2
   + r[1,1] * exp_tau_var_1_3 + r[1,1] * exp_tau_var_1_4
   + r[1,1] * exp_tau_var_1_5 + r[1,1] * exp_tau_var_1_6
   + r[1,1] * exp_tau_var_1_7 ] = 0
 softmax_1_2: - exp_tau_var_1_2 + [ r[1,2] * exp_tau_var_1_0
   + r[1,2] * exp_tau_var_1_1 + r[1,2] * exp_tau_var_1_2
   + r[1,2] * exp_tau_var_1_3 + r[1,2] * exp_tau_var_1_4
   + r[1,2] * exp_tau_var_1_5 + r[1,2] * exp_tau_var_1_6
   + r[1,2] * exp_tau_var_1_7 ] = 0
 softmax_1_3: - exp_tau_var_1_3 + [ r[1,3] * exp_tau_var_1_0
   + r[1,3] * exp_tau_var_1_1 + r[1,3] * exp_tau_var_1_2
   + r[1,3] * exp_tau_var_1_3 + r[1,3] * exp_tau_var_1_4
   + r[1,3] * exp_tau_var_1_5 + r[1,3] * exp_tau_var_1_6
   + r[1,3] * exp_tau_var_1_7 ] = 0
 softmax_1_4: - exp_tau_var_1_4 + [ r[1,4] * exp_tau_var_1_0
   + r[1,4] * exp_tau_var_1_1 + r[1,4] * exp_tau_var_1_2
   + r[1,4] * exp_tau_var_1_3 + r[1,4] * exp_tau_var_1_4
   + r[1,4] * exp_tau_var_1_5 + r[1,4] * exp_tau_var_1_6
   + r[1,4] * exp_tau_var_1_7 ] = 0
 softmax_1_5: - exp_tau_var_1_5 + [ r[1,5] * exp_tau_var_1_0
   + r[1,5] * exp_tau_var_1_1 + r[1,5] * exp_tau_var_1_2
   + r[1,5] * exp_tau_var_1_3 + r[1,5] * exp_tau_var_1_4
   + r[1,5] * exp_tau_var_1_5 + r[1,5] * exp_tau_var_1_6
   + r[1,5] * exp_tau_var_1_7 ] = 0
 softmax_1_6: - exp_tau_var_1_6 + [ r[1,6] * exp_tau_var_1_0
   + r[1,6] * exp_tau_var_1_1 + r[1,6] * exp_tau_var_1_2
   + r[1,6] * exp_tau_var_1_3 + r[1,6] * exp_tau_var_1_4
   + r[1,6] * exp_tau_var_1_5 + r[1,6] * exp_tau_var_1_6
   + r[1,6] * exp_tau_var_1_7 ] = 0
 softmax_1_7: - exp_tau_var_1_7 + [ r[1,7] * exp_tau_var_1_0
   + r[1,7] * exp_tau_var_1_1 + r[1,7] * exp_tau_var_1_2
   + r[1,7] * exp_tau_var_1_3 + r[1,7] * exp_tau_var_1_4
   + r[1,7] * exp_tau_var_1_5 + r[1,7] * exp_tau_var_1_6
   + r[1,7] * exp_tau_var_1_7 ] = 0
 softmax_2_0: - exp_tau_var_2_0 + [ r[2,0] * exp_tau_var_2_0
   + r[2,0] * exp_tau_var_2_1 + r[2,0] * exp_tau_var_2_2
   + r[2,0] * exp_tau_var_2_3 + r[2,0] * exp_tau_var_2_4
   + r[2,0] * exp_tau_var_2_5 + r[2,0] * exp_tau_var_2_6
   + r[2,0] * exp_tau_var_2_7 ] = 0
 softmax_2_1: - exp_tau_var_2_1 + [ r[2,1] * exp_tau_var_2_0
   + r[2,1] * exp_tau_var_2_1 + r[2,1] * exp_tau_var_2_2
   + r[2,1] * exp_tau_var_2_3 + r[2,1] * exp_tau_var_2_4
   + r[2,1] * exp_tau_var_2_5 + r[2,1] * exp_tau_var_2_6
   + r[2,1] * exp_tau_var_2_7 ] = 0
 softmax_2_2: - exp_tau_var_2_2 + [ r[2,2] * exp_tau_var_2_0
   + r[2,2] * exp_tau_var_2_1 + r[2,2] * exp_tau_var_2_2
   + r[2,2] * exp_tau_var_2_3 + r[2,2] * exp_tau_var_2_4
   + r[2,2] * exp_tau_var_2_5 + r[2,2] * exp_tau_var_2_6
   + r[2,2] * exp_tau_var_2_7 ] = 0
 softmax_2_3: - exp_tau_var_2_3 + [ r[2,3] * exp_tau_var_2_0
   + r[2,3] * exp_tau_var_2_1 + r[2,3] * exp_tau_var_2_2
   + r[2,3] * exp_tau_var_2_3 + r[2,3] * exp_tau_var_2_4
   + r[2,3] * exp_tau_var_2_5 + r[2,3] * exp_tau_var_2_6
   + r[2,3] * exp_tau_var_2_7 ] = 0
 softmax_2_4: - exp_tau_var_2_4 + [ r[2,4] * exp_tau_var_2_0
   + r[2,4] * exp_tau_var_2_1 + r[2,4] * exp_tau_var_2_2
   + r[2,4] * exp_tau_var_2_3 + r[2,4] * exp_tau_var_2_4
   + r[2,4] * exp_tau_var_2_5 + r[2,4] * exp_tau_var_2_6
   + r[2,4] * exp_tau_var_2_7 ] = 0
 softmax_2_5: - exp_tau_var_2_5 + [ r[2,5] * exp_tau_var_2_0
   + r[2,5] * exp_tau_var_2_1 + r[2,5] * exp_tau_var_2_2
   + r[2,5] * exp_tau_var_2_3 + r[2,5] * exp_tau_var_2_4
   + r[2,5] * exp_tau_var_2_5 + r[2,5] * exp_tau_var_2_6
   + r[2,5] * exp_tau_var_2_7 ] = 0
 softmax_2_6: - exp_tau_var_2_6 + [ r[2,6] * exp_tau_var_2_0
   + r[2,6] * exp_tau_var_2_1 + r[2,6] * exp_tau_var_2_2
   + r[2,6] * exp_tau_var_2_3 + r[2,6] * exp_tau_var_2_4
   + r[2,6] * exp_tau_var_2_5 + r[2,6] * exp_tau_var_2_6
   + r[2,6] * exp_tau_var_2_7 ] = 0
 softmax_2_7: - exp_tau_var_2_7 + [ r[2,7] * exp_tau_var_2_0
   + r[2,7] * exp_tau_var_2_1 + r[2,7] * exp_tau_var_2_2
   + r[2,7] * exp_tau_var_2_3 + r[2,7] * exp_tau_var_2_4
   + r[2,7] * exp_tau_var_2_5 + r[2,7] * exp_tau_var_2_6
   + r[2,7] * exp_tau_var_2_7 ] = 0
 softmax_3_0: - exp_tau_var_3_0 + [ r[3,0] * exp_tau_var_3_0
   + r[3,0] * exp_tau_var_3_1 + r[3,0] * exp_tau_var_3_2
   + r[3,0] * exp_tau_var_3_3 + r[3,0] * exp_tau_var_3_4
   + r[3,0] * exp_tau_var_3_5 + r[3,0] * exp_tau_var_3_6
   + r[3,0] * exp_tau_var_3_7 ] = 0
 softmax_3_1: - exp_tau_var_3_1 + [ r[3,1] * exp_tau_var_3_0
   + r[3,1] * exp_tau_var_3_1 + r[3,1] * exp_tau_var_3_2
   + r[3,1] * exp_tau_var_3_3 + r[3,1] * exp_tau_var_3_4
   + r[3,1] * exp_tau_var_3_5 + r[3,1] * exp_tau_var_3_6
   + r[3,1] * exp_tau_var_3_7 ] = 0
 softmax_3_2: - exp_tau_var_3_2 + [ r[3,2] * exp_tau_var_3_0
   + r[3,2] * exp_tau_var_3_1 + r[3,2] * exp_tau_var_3_2
   + r[3,2] * exp_tau_var_3_3 + r[3,2] * exp_tau_var_3_4
   + r[3,2] * exp_tau_var_3_5 + r[3,2] * exp_tau_var_3_6
   + r[3,2] * exp_tau_var_3_7 ] = 0
 softmax_3_3: - exp_tau_var_3_3 + [ r[3,3] * exp_tau_var_3_0
   + r[3,3] * exp_tau_var_3_1 + r[3,3] * exp_tau_var_3_2
   + r[3,3] * exp_tau_var_3_3 + r[3,3] * exp_tau_var_3_4
   + r[3,3] * exp_tau_var_3_5 + r[3,3] * exp_tau_var_3_6
   + r[3,3] * exp_tau_var_3_7 ] = 0
 softmax_3_4: - exp_tau_var_3_4 + [ r[3,4] * exp_tau_var_3_0
   + r[3,4] * exp_tau_var_3_1 + r[3,4] * exp_tau_var_3_2
   + r[3,4] * exp_tau_var_3_3 + r[3,4] * exp_tau_var_3_4
   + r[3,4] * exp_tau_var_3_5 + r[3,4] * exp_tau_var_3_6
   + r[3,4] * exp_tau_var_3_7 ] = 0
 softmax_3_5: - exp_tau_var_3_5 + [ r[3,5] * exp_tau_var_3_0
   + r[3,5] * exp_tau_var_3_1 + r[3,5] * exp_tau_var_3_2
   + r[3,5] * exp_tau_var_3_3 + r[3,5] * exp_tau_var_3_4
   + r[3,5] * exp_tau_var_3_5 + r[3,5] * exp_tau_var_3_6
   + r[3,5] * exp_tau_var_3_7 ] = 0
 softmax_3_6: - exp_tau_var_3_6 + [ r[3,6] * exp_tau_var_3_0
   + r[3,6] * exp_tau_var_3_1 + r[3,6] * exp_tau_var_3_2
   + r[3,6] * exp_tau_var_3_3 + r[3,6] * exp_tau_var_3_4
   + r[3,6] * exp_tau_var_3_5 + r[3,6] * exp_tau_var_3_6
   + r[3,6] * exp_tau_var_3_7 ] = 0
 softmax_3_7: - exp_tau_var_3_7 + [ r[3,7] * exp_tau_var_3_0
   + r[3,7] * exp_tau_var_3_1 + r[3,7] * exp_tau_var_3_2
   + r[3,7] * exp_tau_var_3_3 + r[3,7] * exp_tau_var_3_4
   + r[3,7] * exp_tau_var_3_5 + r[3,7] * exp_tau_var_3_6
   + r[3,7] * exp_tau_var_3_7 ] = 0
 softmax_4_0: - exp_tau_var_4_0 + [ r[4,0] * exp_tau_var_4_0
   + r[4,0] * exp_tau_var_4_1 + r[4,0] * exp_tau_var_4_2
   + r[4,0] * exp_tau_var_4_3 + r[4,0] * exp_tau_var_4_4
   + r[4,0] * exp_tau_var_4_5 + r[4,0] * exp_tau_var_4_6
   + r[4,0] * exp_tau_var_4_7 ] = 0
 softmax_4_1: - exp_tau_var_4_1 + [ r[4,1] * exp_tau_var_4_0
   + r[4,1] * exp_tau_var_4_1 + r[4,1] * exp_tau_var_4_2
   + r[4,1] * exp_tau_var_4_3 + r[4,1] * exp_tau_var_4_4
   + r[4,1] * exp_tau_var_4_5 + r[4,1] * exp_tau_var_4_6
   + r[4,1] * exp_tau_var_4_7 ] = 0
 softmax_4_2: - exp_tau_var_4_2 + [ r[4,2] * exp_tau_var_4_0
   + r[4,2] * exp_tau_var_4_1 + r[4,2] * exp_tau_var_4_2
   + r[4,2] * exp_tau_var_4_3 + r[4,2] * exp_tau_var_4_4
   + r[4,2] * exp_tau_var_4_5 + r[4,2] * exp_tau_var_4_6
   + r[4,2] * exp_tau_var_4_7 ] = 0
 softmax_4_3: - exp_tau_var_4_3 + [ r[4,3] * exp_tau_var_4_0
   + r[4,3] * exp_tau_var_4_1 + r[4,3] * exp_tau_var_4_2
   + r[4,3] * exp_tau_var_4_3 + r[4,3] * exp_tau_var_4_4
   + r[4,3] * exp_tau_var_4_5 + r[4,3] * exp_tau_var_4_6
   + r[4,3] * exp_tau_var_4_7 ] = 0
 softmax_4_4: - exp_tau_var_4_4 + [ r[4,4] * exp_tau_var_4_0
   + r[4,4] * exp_tau_var_4_1 + r[4,4] * exp_tau_var_4_2
   + r[4,4] * exp_tau_var_4_3 + r[4,4] * exp_tau_var_4_4
   + r[4,4] * exp_tau_var_4_5 + r[4,4] * exp_tau_var_4_6
   + r[4,4] * exp_tau_var_4_7 ] = 0
 softmax_4_5: - exp_tau_var_4_5 + [ r[4,5] * exp_tau_var_4_0
   + r[4,5] * exp_tau_var_4_1 + r[4,5] * exp_tau_var_4_2
   + r[4,5] * exp_tau_var_4_3 + r[4,5] * exp_tau_var_4_4
   + r[4,5] * exp_tau_var_4_5 + r[4,5] * exp_tau_var_4_6
   + r[4,5] * exp_tau_var_4_7 ] = 0
 softmax_4_6: - exp_tau_var_4_6 + [ r[4,6] * exp_tau_var_4_0
   + r[4,6] * exp_tau_var_4_1 + r[4,6] * exp_tau_var_4_2
   + r[4,6] * exp_tau_var_4_3 + r[4,6] * exp_tau_var_4_4
   + r[4,6] * exp_tau_var_4_5 + r[4,6] * exp_tau_var_4_6
   + r[4,6] * exp_tau_var_4_7 ] = 0
 softmax_4_7: - exp_tau_var_4_7 + [ r[4,7] * exp_tau_var_4_0
   + r[4,7] * exp_tau_var_4_1 + r[4,7] * exp_tau_var_4_2
   + r[4,7] * exp_tau_var_4_3 + r[4,7] * exp_tau_var_4_4
   + r[4,7] * exp_tau_var_4_5 + r[4,7] * exp_tau_var_4_6
   + r[4,7] * exp_tau_var_4_7 ] = 0
 softmax_5_0: - exp_tau_var_5_0 + [ r[5,0] * exp_tau_var_5_0
   + r[5,0] * exp_tau_var_5_1 + r[5,0] * exp_tau_var_5_2
   + r[5,0] * exp_tau_var_5_3 + r[5,0] * exp_tau_var_5_4
   + r[5,0] * exp_tau_var_5_5 + r[5,0] * exp_tau_var_5_6
   + r[5,0] * exp_tau_var_5_7 ] = 0
 softmax_5_1: - exp_tau_var_5_1 + [ r[5,1] * exp_tau_var_5_0
   + r[5,1] * exp_tau_var_5_1 + r[5,1] * exp_tau_var_5_2
   + r[5,1] * exp_tau_var_5_3 + r[5,1] * exp_tau_var_5_4
   + r[5,1] * exp_tau_var_5_5 + r[5,1] * exp_tau_var_5_6
   + r[5,1] * exp_tau_var_5_7 ] = 0
 softmax_5_2: - exp_tau_var_5_2 + [ r[5,2] * exp_tau_var_5_0
   + r[5,2] * exp_tau_var_5_1 + r[5,2] * exp_tau_var_5_2
   + r[5,2] * exp_tau_var_5_3 + r[5,2] * exp_tau_var_5_4
   + r[5,2] * exp_tau_var_5_5 + r[5,2] * exp_tau_var_5_6
   + r[5,2] * exp_tau_var_5_7 ] = 0
 softmax_5_3: - exp_tau_var_5_3 + [ r[5,3] * exp_tau_var_5_0
   + r[5,3] * exp_tau_var_5_1 + r[5,3] * exp_tau_var_5_2
   + r[5,3] * exp_tau_var_5_3 + r[5,3] * exp_tau_var_5_4
   + r[5,3] * exp_tau_var_5_5 + r[5,3] * exp_tau_var_5_6
   + r[5,3] * exp_tau_var_5_7 ] = 0
 softmax_5_4: - exp_tau_var_5_4 + [ r[5,4] * exp_tau_var_5_0
   + r[5,4] * exp_tau_var_5_1 + r[5,4] * exp_tau_var_5_2
   + r[5,4] * exp_tau_var_5_3 + r[5,4] * exp_tau_var_5_4
   + r[5,4] * exp_tau_var_5_5 + r[5,4] * exp_tau_var_5_6
   + r[5,4] * exp_tau_var_5_7 ] = 0
 softmax_5_5: - exp_tau_var_5_5 + [ r[5,5] * exp_tau_var_5_0
   + r[5,5] * exp_tau_var_5_1 + r[5,5] * exp_tau_var_5_2
   + r[5,5] * exp_tau_var_5_3 + r[5,5] * exp_tau_var_5_4
   + r[5,5] * exp_tau_var_5_5 + r[5,5] * exp_tau_var_5_6
   + r[5,5] * exp_tau_var_5_7 ] = 0
 softmax_5_6: - exp_tau_var_5_6 + [ r[5,6] * exp_tau_var_5_0
   + r[5,6] * exp_tau_var_5_1 + r[5,6] * exp_tau_var_5_2
   + r[5,6] * exp_tau_var_5_3 + r[5,6] * exp_tau_var_5_4
   + r[5,6] * exp_tau_var_5_5 + r[5,6] * exp_tau_var_5_6
   + r[5,6] * exp_tau_var_5_7 ] = 0
 softmax_5_7: - exp_tau_var_5_7 + [ r[5,7] * exp_tau_var_5_0
   + r[5,7] * exp_tau_var_5_1 + r[5,7] * exp_tau_var_5_2
   + r[5,7] * exp_tau_var_5_3 + r[5,7] * exp_tau_var_5_4
   + r[5,7] * exp_tau_var_5_5 + r[5,7] * exp_tau_var_5_6
   + r[5,7] * exp_tau_var_5_7 ] = 0
 softmax_6_0: - exp_tau_var_6_0 + [ r[6,0] * exp_tau_var_6_0
   + r[6,0] * exp_tau_var_6_1 + r[6,0] * exp_tau_var_6_2
   + r[6,0] * exp_tau_var_6_3 + r[6,0] * exp_tau_var_6_4
   + r[6,0] * exp_tau_var_6_5 + r[6,0] * exp_tau_var_6_6
   + r[6,0] * exp_tau_var_6_7 ] = 0
 softmax_6_1: - exp_tau_var_6_1 + [ r[6,1] * exp_tau_var_6_0
   + r[6,1] * exp_tau_var_6_1 + r[6,1] * exp_tau_var_6_2
   + r[6,1] * exp_tau_var_6_3 + r[6,1] * exp_tau_var_6_4
   + r[6,1] * exp_tau_var_6_5 + r[6,1] * exp_tau_var_6_6
   + r[6,1] * exp_tau_var_6_7 ] = 0
 softmax_6_2: - exp_tau_var_6_2 + [ r[6,2] * exp_tau_var_6_0
   + r[6,2] * exp_tau_var_6_1 + r[6,2] * exp_tau_var_6_2
   + r[6,2] * exp_tau_var_6_3 + r[6,2] * exp_tau_var_6_4
   + r[6,2] * exp_tau_var_6_5 + r[6,2] * exp_tau_var_6_6
   + r[6,2] * exp_tau_var_6_7 ] = 0
 softmax_6_3: - exp_tau_var_6_3 + [ r[6,3] * exp_tau_var_6_0
   + r[6,3] * exp_tau_var_6_1 + r[6,3] * exp_tau_var_6_2
   + r[6,3] * exp_tau_var_6_3 + r[6,3] * exp_tau_var_6_4
   + r[6,3] * exp_tau_var_6_5 + r[6,3] * exp_tau_var_6_6
   + r[6,3] * exp_tau_var_6_7 ] = 0
 softmax_6_4: - exp_tau_var_6_4 + [ r[6,4] * exp_tau_var_6_0
   + r[6,4] * exp_tau_var_6_1 + r[6,4] * exp_tau_var_6_2
   + r[6,4] * exp_tau_var_6_3 + r[6,4] * exp_tau_var_6_4
   + r[6,4] * exp_tau_var_6_5 + r[6,4] * exp_tau_var_6_6
   + r[6,4] * exp_tau_var_6_7 ] = 0
 softmax_6_5: - exp_tau_var_6_5 + [ r[6,5] * exp_tau_var_6_0
   + r[6,5] * exp_tau_var_6_1 + r[6,5] * exp_tau_var_6_2
   + r[6,5] * exp_tau_var_6_3 + r[6,5] * exp_tau_var_6_4
   + r[6,5] * exp_tau_var_6_5 + r[6,5] * exp_tau_var_6_6
   + r[6,5] * exp_tau_var_6_7 ] = 0
 softmax_6_6: - exp_tau_var_6_6 + [ r[6,6] * exp_tau_var_6_0
   + r[6,6] * exp_tau_var_6_1 + r[6,6] * exp_tau_var_6_2
   + r[6,6] * exp_tau_var_6_3 + r[6,6] * exp_tau_var_6_4
   + r[6,6] * exp_tau_var_6_5 + r[6,6] * exp_tau_var_6_6
   + r[6,6] * exp_tau_var_6_7 ] = 0
 softmax_6_7: - exp_tau_var_6_7 + [ r[6,7] * exp_tau_var_6_0
   + r[6,7] * exp_tau_var_6_1 + r[6,7] * exp_tau_var_6_2
   + r[6,7] * exp_tau_var_6_3 + r[6,7] * exp_tau_var_6_4
   + r[6,7] * exp_tau_var_6_5 + r[6,7] * exp_tau_var_6_6
   + r[6,7] * exp_tau_var_6_7 ] = 0
 softmax_7_0: - exp_tau_var_7_0 + [ r[7,0] * exp_tau_var_7_0
   + r[7,0] * exp_tau_var_7_1 + r[7,0] * exp_tau_var_7_2
   + r[7,0] * exp_tau_var_7_3 + r[7,0] * exp_tau_var_7_4
   + r[7,0] * exp_tau_var_7_5 + r[7,0] * exp_tau_var_7_6
   + r[7,0] * exp_tau_var_7_7 ] = 0
 softmax_7_1: - exp_tau_var_7_1 + [ r[7,1] * exp_tau_var_7_0
   + r[7,1] * exp_tau_var_7_1 + r[7,1] * exp_tau_var_7_2
   + r[7,1] * exp_tau_var_7_3 + r[7,1] * exp_tau_var_7_4
   + r[7,1] * exp_tau_var_7_5 + r[7,1] * exp_tau_var_7_6
   + r[7,1] * exp_tau_var_7_7 ] = 0
 softmax_7_2: - exp_tau_var_7_2 + [ r[7,2] * exp_tau_var_7_0
   + r[7,2] * exp_tau_var_7_1 + r[7,2] * exp_tau_var_7_2
   + r[7,2] * exp_tau_var_7_3 + r[7,2] * exp_tau_var_7_4
   + r[7,2] * exp_tau_var_7_5 + r[7,2] * exp_tau_var_7_6
   + r[7,2] * exp_tau_var_7_7 ] = 0
 softmax_7_3: - exp_tau_var_7_3 + [ r[7,3] * exp_tau_var_7_0
   + r[7,3] * exp_tau_var_7_1 + r[7,3] * exp_tau_var_7_2
   + r[7,3] * exp_tau_var_7_3 + r[7,3] * exp_tau_var_7_4
   + r[7,3] * exp_tau_var_7_5 + r[7,3] * exp_tau_var_7_6
   + r[7,3] * exp_tau_var_7_7 ] = 0
 softmax_7_4: - exp_tau_var_7_4 + [ r[7,4] * exp_tau_var_7_0
   + r[7,4] * exp_tau_var_7_1 + r[7,4] * exp_tau_var_7_2
   + r[7,4] * exp_tau_var_7_3 + r[7,4] * exp_tau_var_7_4
   + r[7,4] * exp_tau_var_7_5 + r[7,4] * exp_tau_var_7_6
   + r[7,4] * exp_tau_var_7_7 ] = 0
 softmax_7_5: - exp_tau_var_7_5 + [ r[7,5] * exp_tau_var_7_0
   + r[7,5] * exp_tau_var_7_1 + r[7,5] * exp_tau_var_7_2
   + r[7,5] * exp_tau_var_7_3 + r[7,5] * exp_tau_var_7_4
   + r[7,5] * exp_tau_var_7_5 + r[7,5] * exp_tau_var_7_6
   + r[7,5] * exp_tau_var_7_7 ] = 0
 softmax_7_6: - exp_tau_var_7_6 + [ r[7,6] * exp_tau_var_7_0
   + r[7,6] * exp_tau_var_7_1 + r[7,6] * exp_tau_var_7_2
   + r[7,6] * exp_tau_var_7_3 + r[7,6] * exp_tau_var_7_4
   + r[7,6] * exp_tau_var_7_5 + r[7,6] * exp_tau_var_7_6
   + r[7,6] * exp_tau_var_7_7 ] = 0
 softmax_7_7: - exp_tau_var_7_7 + [ r[7,7] * exp_tau_var_7_0
   + r[7,7] * exp_tau_var_7_1 + r[7,7] * exp_tau_var_7_2
   + r[7,7] * exp_tau_var_7_3 + r[7,7] * exp_tau_var_7_4
   + r[7,7] * exp_tau_var_7_5 + r[7,7] * exp_tau_var_7_6
   + r[7,7] * exp_tau_var_7_7 ] = 0
 softmax_8_0: - exp_tau_var_8_0 + [ r[8,0] * exp_tau_var_8_0
   + r[8,0] * exp_tau_var_8_1 + r[8,0] * exp_tau_var_8_2
   + r[8,0] * exp_tau_var_8_3 + r[8,0] * exp_tau_var_8_4
   + r[8,0] * exp_tau_var_8_5 + r[8,0] * exp_tau_var_8_6
   + r[8,0] * exp_tau_var_8_7 ] = 0
 softmax_8_1: - exp_tau_var_8_1 + [ r[8,1] * exp_tau_var_8_0
   + r[8,1] * exp_tau_var_8_1 + r[8,1] * exp_tau_var_8_2
   + r[8,1] * exp_tau_var_8_3 + r[8,1] * exp_tau_var_8_4
   + r[8,1] * exp_tau_var_8_5 + r[8,1] * exp_tau_var_8_6
   + r[8,1] * exp_tau_var_8_7 ] = 0
 softmax_8_2: - exp_tau_var_8_2 + [ r[8,2] * exp_tau_var_8_0
   + r[8,2] * exp_tau_var_8_1 + r[8,2] * exp_tau_var_8_2
   + r[8,2] * exp_tau_var_8_3 + r[8,2] * exp_tau_var_8_4
   + r[8,2] * exp_tau_var_8_5 + r[8,2] * exp_tau_var_8_6
   + r[8,2] * exp_tau_var_8_7 ] = 0
 softmax_8_3: - exp_tau_var_8_3 + [ r[8,3] * exp_tau_var_8_0
   + r[8,3] * exp_tau_var_8_1 + r[8,3] * exp_tau_var_8_2
   + r[8,3] * exp_tau_var_8_3 + r[8,3] * exp_tau_var_8_4
   + r[8,3] * exp_tau_var_8_5 + r[8,3] * exp_tau_var_8_6
   + r[8,3] * exp_tau_var_8_7 ] = 0
 softmax_8_4: - exp_tau_var_8_4 + [ r[8,4] * exp_tau_var_8_0
   + r[8,4] * exp_tau_var_8_1 + r[8,4] * exp_tau_var_8_2
   + r[8,4] * exp_tau_var_8_3 + r[8,4] * exp_tau_var_8_4
   + r[8,4] * exp_tau_var_8_5 + r[8,4] * exp_tau_var_8_6
   + r[8,4] * exp_tau_var_8_7 ] = 0
 softmax_8_5: - exp_tau_var_8_5 + [ r[8,5] * exp_tau_var_8_0
   + r[8,5] * exp_tau_var_8_1 + r[8,5] * exp_tau_var_8_2
   + r[8,5] * exp_tau_var_8_3 + r[8,5] * exp_tau_var_8_4
   + r[8,5] * exp_tau_var_8_5 + r[8,5] * exp_tau_var_8_6
   + r[8,5] * exp_tau_var_8_7 ] = 0
 softmax_8_6: - exp_tau_var_8_6 + [ r[8,6] * exp_tau_var_8_0
   + r[8,6] * exp_tau_var_8_1 + r[8,6] * exp_tau_var_8_2
   + r[8,6] * exp_tau_var_8_3 + r[8,6] * exp_tau_var_8_4
   + r[8,6] * exp_tau_var_8_5 + r[8,6] * exp_tau_var_8_6
   + r[8,6] * exp_tau_var_8_7 ] = 0
 softmax_8_7: - exp_tau_var_8_7 + [ r[8,7] * exp_tau_var_8_0
   + r[8,7] * exp_tau_var_8_1 + r[8,7] * exp_tau_var_8_2
   + r[8,7] * exp_tau_var_8_3 + r[8,7] * exp_tau_var_8_4
   + r[8,7] * exp_tau_var_8_5 + r[8,7] * exp_tau_var_8_6
   + r[8,7] * exp_tau_var_8_7 ] = 0
 softmax_9_0: - exp_tau_var_9_0 + [ r[9,0] * exp_tau_var_9_0
   + r[9,0] * exp_tau_var_9_1 + r[9,0] * exp_tau_var_9_2
   + r[9,0] * exp_tau_var_9_3 + r[9,0] * exp_tau_var_9_4
   + r[9,0] * exp_tau_var_9_5 + r[9,0] * exp_tau_var_9_6
   + r[9,0] * exp_tau_var_9_7 ] = 0
 softmax_9_1: - exp_tau_var_9_1 + [ r[9,1] * exp_tau_var_9_0
   + r[9,1] * exp_tau_var_9_1 + r[9,1] * exp_tau_var_9_2
   + r[9,1] * exp_tau_var_9_3 + r[9,1] * exp_tau_var_9_4
   + r[9,1] * exp_tau_var_9_5 + r[9,1] * exp_tau_var_9_6
   + r[9,1] * exp_tau_var_9_7 ] = 0
 softmax_9_2: - exp_tau_var_9_2 + [ r[9,2] * exp_tau_var_9_0
   + r[9,2] * exp_tau_var_9_1 + r[9,2] * exp_tau_var_9_2
   + r[9,2] * exp_tau_var_9_3 + r[9,2] * exp_tau_var_9_4
   + r[9,2] * exp_tau_var_9_5 + r[9,2] * exp_tau_var_9_6
   + r[9,2] * exp_tau_var_9_7 ] = 0
 softmax_9_3: - exp_tau_var_9_3 + [ r[9,3] * exp_tau_var_9_0
   + r[9,3] * exp_tau_var_9_1 + r[9,3] * exp_tau_var_9_2
   + r[9,3] * exp_tau_var_9_3 + r[9,3] * exp_tau_var_9_4
   + r[9,3] * exp_tau_var_9_5 + r[9,3] * exp_tau_var_9_6
   + r[9,3] * exp_tau_var_9_7 ] = 0
 softmax_9_4: - exp_tau_var_9_4 + [ r[9,4] * exp_tau_var_9_0
   + r[9,4] * exp_tau_var_9_1 + r[9,4] * exp_tau_var_9_2
   + r[9,4] * exp_tau_var_9_3 + r[9,4] * exp_tau_var_9_4
   + r[9,4] * exp_tau_var_9_5 + r[9,4] * exp_tau_var_9_6
   + r[9,4] * exp_tau_var_9_7 ] = 0
 softmax_9_5: - exp_tau_var_9_5 + [ r[9,5] * exp_tau_var_9_0
   + r[9,5] * exp_tau_var_9_1 + r[9,5] * exp_tau_var_9_2
   + r[9,5] * exp_tau_var_9_3 + r[9,5] * exp_tau_var_9_4
   + r[9,5] * exp_tau_var_9_5 + r[9,5] * exp_tau_var_9_6
   + r[9,5] * exp_tau_var_9_7 ] = 0
 softmax_9_6: - exp_tau_var_9_6 + [ r[9,6] * exp_tau_var_9_0
   + r[9,6] * exp_tau_var_9_1 + r[9,6] * exp_tau_var_9_2
   + r[9,6] * exp_tau_var_9_3 + r[9,6] * exp_tau_var_9_4
   + r[9,6] * exp_tau_var_9_5 + r[9,6] * exp_tau_var_9_6
   + r[9,6] * exp_tau_var_9_7 ] = 0
 softmax_9_7: - exp_tau_var_9_7 + [ r[9,7] * exp_tau_var_9_0
   + r[9,7] * exp_tau_var_9_1 + r[9,7] * exp_tau_var_9_2
   + r[9,7] * exp_tau_var_9_3 + r[9,7] * exp_tau_var_9_4
   + r[9,7] * exp_tau_var_9_5 + r[9,7] * exp_tau_var_9_6
   + r[9,7] * exp_tau_var_9_7 ] = 0
 softmax_10_0: - exp_tau_var_10_0 + [ r[10,0] * exp_tau_var_10_0
   + r[10,0] * exp_tau_var_10_1 + r[10,0] * exp_tau_var_10_2
   + r[10,0] * exp_tau_var_10_3 + r[10,0] * exp_tau_var_10_4
   + r[10,0] * exp_tau_var_10_5 + r[10,0] * exp_tau_var_10_6
   + r[10,0] * exp_tau_var_10_7 ] = 0
 softmax_10_1: - exp_tau_var_10_1 + [ r[10,1] * exp_tau_var_10_0
   + r[10,1] * exp_tau_var_10_1 + r[10,1] * exp_tau_var_10_2
   + r[10,1] * exp_tau_var_10_3 + r[10,1] * exp_tau_var_10_4
   + r[10,1] * exp_tau_var_10_5 + r[10,1] * exp_tau_var_10_6
   + r[10,1] * exp_tau_var_10_7 ] = 0
 softmax_10_2: - exp_tau_var_10_2 + [ r[10,2] * exp_tau_var_10_0
   + r[10,2] * exp_tau_var_10_1 + r[10,2] * exp_tau_var_10_2
   + r[10,2] * exp_tau_var_10_3 + r[10,2] * exp_tau_var_10_4
   + r[10,2] * exp_tau_var_10_5 + r[10,2] * exp_tau_var_10_6
   + r[10,2] * exp_tau_var_10_7 ] = 0
 softmax_10_3: - exp_tau_var_10_3 + [ r[10,3] * exp_tau_var_10_0
   + r[10,3] * exp_tau_var_10_1 + r[10,3] * exp_tau_var_10_2
   + r[10,3] * exp_tau_var_10_3 + r[10,3] * exp_tau_var_10_4
   + r[10,3] * exp_tau_var_10_5 + r[10,3] * exp_tau_var_10_6
   + r[10,3] * exp_tau_var_10_7 ] = 0
 softmax_10_4: - exp_tau_var_10_4 + [ r[10,4] * exp_tau_var_10_0
   + r[10,4] * exp_tau_var_10_1 + r[10,4] * exp_tau_var_10_2
   + r[10,4] * exp_tau_var_10_3 + r[10,4] * exp_tau_var_10_4
   + r[10,4] * exp_tau_var_10_5 + r[10,4] * exp_tau_var_10_6
   + r[10,4] * exp_tau_var_10_7 ] = 0
 softmax_10_5: - exp_tau_var_10_5 + [ r[10,5] * exp_tau_var_10_0
   + r[10,5] * exp_tau_var_10_1 + r[10,5] * exp_tau_var_10_2
   + r[10,5] * exp_tau_var_10_3 + r[10,5] * exp_tau_var_10_4
   + r[10,5] * exp_tau_var_10_5 + r[10,5] * exp_tau_var_10_6
   + r[10,5] * exp_tau_var_10_7 ] = 0
 softmax_10_6: - exp_tau_var_10_6 + [ r[10,6] * exp_tau_var_10_0
   + r[10,6] * exp_tau_var_10_1 + r[10,6] * exp_tau_var_10_2
   + r[10,6] * exp_tau_var_10_3 + r[10,6] * exp_tau_var_10_4
   + r[10,6] * exp_tau_var_10_5 + r[10,6] * exp_tau_var_10_6
   + r[10,6] * exp_tau_var_10_7 ] = 0
 softmax_10_7: - exp_tau_var_10_7 + [ r[10,7] * exp_tau_var_10_0
   + r[10,7] * exp_tau_var_10_1 + r[10,7] * exp_tau_var_10_2
   + r[10,7] * exp_tau_var_10_3 + r[10,7] * exp_tau_var_10_4
   + r[10,7] * exp_tau_var_10_5 + r[10,7] * exp_tau_var_10_6
   + r[10,7] * exp_tau_var_10_7 ] = 0
 softmax_11_0: - exp_tau_var_11_0 + [ r[11,0] * exp_tau_var_11_0
   + r[11,0] * exp_tau_var_11_1 + r[11,0] * exp_tau_var_11_2
   + r[11,0] * exp_tau_var_11_3 + r[11,0] * exp_tau_var_11_4
   + r[11,0] * exp_tau_var_11_5 + r[11,0] * exp_tau_var_11_6
   + r[11,0] * exp_tau_var_11_7 ] = 0
 softmax_11_1: - exp_tau_var_11_1 + [ r[11,1] * exp_tau_var_11_0
   + r[11,1] * exp_tau_var_11_1 + r[11,1] * exp_tau_var_11_2
   + r[11,1] * exp_tau_var_11_3 + r[11,1] * exp_tau_var_11_4
   + r[11,1] * exp_tau_var_11_5 + r[11,1] * exp_tau_var_11_6
   + r[11,1] * exp_tau_var_11_7 ] = 0
 softmax_11_2: - exp_tau_var_11_2 + [ r[11,2] * exp_tau_var_11_0
   + r[11,2] * exp_tau_var_11_1 + r[11,2] * exp_tau_var_11_2
   + r[11,2] * exp_tau_var_11_3 + r[11,2] * exp_tau_var_11_4
   + r[11,2] * exp_tau_var_11_5 + r[11,2] * exp_tau_var_11_6
   + r[11,2] * exp_tau_var_11_7 ] = 0
 softmax_11_3: - exp_tau_var_11_3 + [ r[11,3] * exp_tau_var_11_0
   + r[11,3] * exp_tau_var_11_1 + r[11,3] * exp_tau_var_11_2
   + r[11,3] * exp_tau_var_11_3 + r[11,3] * exp_tau_var_11_4
   + r[11,3] * exp_tau_var_11_5 + r[11,3] * exp_tau_var_11_6
   + r[11,3] * exp_tau_var_11_7 ] = 0
 softmax_11_4: - exp_tau_var_11_4 + [ r[11,4] * exp_tau_var_11_0
   + r[11,4] * exp_tau_var_11_1 + r[11,4] * exp_tau_var_11_2
   + r[11,4] * exp_tau_var_11_3 + r[11,4] * exp_tau_var_11_4
   + r[11,4] * exp_tau_var_11_5 + r[11,4] * exp_tau_var_11_6
   + r[11,4] * exp_tau_var_11_7 ] = 0
 softmax_11_5: - exp_tau_var_11_5 + [ r[11,5] * exp_tau_var_11_0
   + r[11,5] * exp_tau_var_11_1 + r[11,5] * exp_tau_var_11_2
   + r[11,5] * exp_tau_var_11_3 + r[11,5] * exp_tau_var_11_4
   + r[11,5] * exp_tau_var_11_5 + r[11,5] * exp_tau_var_11_6
   + r[11,5] * exp_tau_var_11_7 ] = 0
 softmax_11_6: - exp_tau_var_11_6 + [ r[11,6] * exp_tau_var_11_0
   + r[11,6] * exp_tau_var_11_1 + r[11,6] * exp_tau_var_11_2
   + r[11,6] * exp_tau_var_11_3 + r[11,6] * exp_tau_var_11_4
   + r[11,6] * exp_tau_var_11_5 + r[11,6] * exp_tau_var_11_6
   + r[11,6] * exp_tau_var_11_7 ] = 0
 softmax_11_7: - exp_tau_var_11_7 + [ r[11,7] * exp_tau_var_11_0
   + r[11,7] * exp_tau_var_11_1 + r[11,7] * exp_tau_var_11_2
   + r[11,7] * exp_tau_var_11_3 + r[11,7] * exp_tau_var_11_4
   + r[11,7] * exp_tau_var_11_5 + r[11,7] * exp_tau_var_11_6
   + r[11,7] * exp_tau_var_11_7 ] = 0
 softmax_12_0: - exp_tau_var_12_0 + [ r[12,0] * exp_tau_var_12_0
   + r[12,0] * exp_tau_var_12_1 + r[12,0] * exp_tau_var_12_2
   + r[12,0] * exp_tau_var_12_3 + r[12,0] * exp_tau_var_12_4
   + r[12,0] * exp_tau_var_12_5 + r[12,0] * exp_tau_var_12_6
   + r[12,0] * exp_tau_var_12_7 ] = 0
 softmax_12_1: - exp_tau_var_12_1 + [ r[12,1] * exp_tau_var_12_0
   + r[12,1] * exp_tau_var_12_1 + r[12,1] * exp_tau_var_12_2
   + r[12,1] * exp_tau_var_12_3 + r[12,1] * exp_tau_var_12_4
   + r[12,1] * exp_tau_var_12_5 + r[12,1] * exp_tau_var_12_6
   + r[12,1] * exp_tau_var_12_7 ] = 0
 softmax_12_2: - exp_tau_var_12_2 + [ r[12,2] * exp_tau_var_12_0
   + r[12,2] * exp_tau_var_12_1 + r[12,2] * exp_tau_var_12_2
   + r[12,2] * exp_tau_var_12_3 + r[12,2] * exp_tau_var_12_4
   + r[12,2] * exp_tau_var_12_5 + r[12,2] * exp_tau_var_12_6
   + r[12,2] * exp_tau_var_12_7 ] = 0
 softmax_12_3: - exp_tau_var_12_3 + [ r[12,3] * exp_tau_var_12_0
   + r[12,3] * exp_tau_var_12_1 + r[12,3] * exp_tau_var_12_2
   + r[12,3] * exp_tau_var_12_3 + r[12,3] * exp_tau_var_12_4
   + r[12,3] * exp_tau_var_12_5 + r[12,3] * exp_tau_var_12_6
   + r[12,3] * exp_tau_var_12_7 ] = 0
 softmax_12_4: - exp_tau_var_12_4 + [ r[12,4] * exp_tau_var_12_0
   + r[12,4] * exp_tau_var_12_1 + r[12,4] * exp_tau_var_12_2
   + r[12,4] * exp_tau_var_12_3 + r[12,4] * exp_tau_var_12_4
   + r[12,4] * exp_tau_var_12_5 + r[12,4] * exp_tau_var_12_6
   + r[12,4] * exp_tau_var_12_7 ] = 0
 softmax_12_5: - exp_tau_var_12_5 + [ r[12,5] * exp_tau_var_12_0
   + r[12,5] * exp_tau_var_12_1 + r[12,5] * exp_tau_var_12_2
   + r[12,5] * exp_tau_var_12_3 + r[12,5] * exp_tau_var_12_4
   + r[12,5] * exp_tau_var_12_5 + r[12,5] * exp_tau_var_12_6
   + r[12,5] * exp_tau_var_12_7 ] = 0
 softmax_12_6: - exp_tau_var_12_6 + [ r[12,6] * exp_tau_var_12_0
   + r[12,6] * exp_tau_var_12_1 + r[12,6] * exp_tau_var_12_2
   + r[12,6] * exp_tau_var_12_3 + r[12,6] * exp_tau_var_12_4
   + r[12,6] * exp_tau_var_12_5 + r[12,6] * exp_tau_var_12_6
   + r[12,6] * exp_tau_var_12_7 ] = 0
 softmax_12_7: - exp_tau_var_12_7 + [ r[12,7] * exp_tau_var_12_0
   + r[12,7] * exp_tau_var_12_1 + r[12,7] * exp_tau_var_12_2
   + r[12,7] * exp_tau_var_12_3 + r[12,7] * exp_tau_var_12_4
   + r[12,7] * exp_tau_var_12_5 + r[12,7] * exp_tau_var_12_6
   + r[12,7] * exp_tau_var_12_7 ] = 0
 softmax_13_0: - exp_tau_var_13_0 + [ r[13,0] * exp_tau_var_13_0
   + r[13,0] * exp_tau_var_13_1 + r[13,0] * exp_tau_var_13_2
   + r[13,0] * exp_tau_var_13_3 + r[13,0] * exp_tau_var_13_4
   + r[13,0] * exp_tau_var_13_5 + r[13,0] * exp_tau_var_13_6
   + r[13,0] * exp_tau_var_13_7 ] = 0
 softmax_13_1: - exp_tau_var_13_1 + [ r[13,1] * exp_tau_var_13_0
   + r[13,1] * exp_tau_var_13_1 + r[13,1] * exp_tau_var_13_2
   + r[13,1] * exp_tau_var_13_3 + r[13,1] * exp_tau_var_13_4
   + r[13,1] * exp_tau_var_13_5 + r[13,1] * exp_tau_var_13_6
   + r[13,1] * exp_tau_var_13_7 ] = 0
 softmax_13_2: - exp_tau_var_13_2 + [ r[13,2] * exp_tau_var_13_0
   + r[13,2] * exp_tau_var_13_1 + r[13,2] * exp_tau_var_13_2
   + r[13,2] * exp_tau_var_13_3 + r[13,2] * exp_tau_var_13_4
   + r[13,2] * exp_tau_var_13_5 + r[13,2] * exp_tau_var_13_6
   + r[13,2] * exp_tau_var_13_7 ] = 0
 softmax_13_3: - exp_tau_var_13_3 + [ r[13,3] * exp_tau_var_13_0
   + r[13,3] * exp_tau_var_13_1 + r[13,3] * exp_tau_var_13_2
   + r[13,3] * exp_tau_var_13_3 + r[13,3] * exp_tau_var_13_4
   + r[13,3] * exp_tau_var_13_5 + r[13,3] * exp_tau_var_13_6
   + r[13,3] * exp_tau_var_13_7 ] = 0
 softmax_13_4: - exp_tau_var_13_4 + [ r[13,4] * exp_tau_var_13_0
   + r[13,4] * exp_tau_var_13_1 + r[13,4] * exp_tau_var_13_2
   + r[13,4] * exp_tau_var_13_3 + r[13,4] * exp_tau_var_13_4
   + r[13,4] * exp_tau_var_13_5 + r[13,4] * exp_tau_var_13_6
   + r[13,4] * exp_tau_var_13_7 ] = 0
 softmax_13_5: - exp_tau_var_13_5 + [ r[13,5] * exp_tau_var_13_0
   + r[13,5] * exp_tau_var_13_1 + r[13,5] * exp_tau_var_13_2
   + r[13,5] * exp_tau_var_13_3 + r[13,5] * exp_tau_var_13_4
   + r[13,5] * exp_tau_var_13_5 + r[13,5] * exp_tau_var_13_6
   + r[13,5] * exp_tau_var_13_7 ] = 0
 softmax_13_6: - exp_tau_var_13_6 + [ r[13,6] * exp_tau_var_13_0
   + r[13,6] * exp_tau_var_13_1 + r[13,6] * exp_tau_var_13_2
   + r[13,6] * exp_tau_var_13_3 + r[13,6] * exp_tau_var_13_4
   + r[13,6] * exp_tau_var_13_5 + r[13,6] * exp_tau_var_13_6
   + r[13,6] * exp_tau_var_13_7 ] = 0
 softmax_13_7: - exp_tau_var_13_7 + [ r[13,7] * exp_tau_var_13_0
   + r[13,7] * exp_tau_var_13_1 + r[13,7] * exp_tau_var_13_2
   + r[13,7] * exp_tau_var_13_3 + r[13,7] * exp_tau_var_13_4
   + r[13,7] * exp_tau_var_13_5 + r[13,7] * exp_tau_var_13_6
   + r[13,7] * exp_tau_var_13_7 ] = 0
 softmax_14_0: - exp_tau_var_14_0 + [ r[14,0] * exp_tau_var_14_0
   + r[14,0] * exp_tau_var_14_1 + r[14,0] * exp_tau_var_14_2
   + r[14,0] * exp_tau_var_14_3 + r[14,0] * exp_tau_var_14_4
   + r[14,0] * exp_tau_var_14_5 + r[14,0] * exp_tau_var_14_6
   + r[14,0] * exp_tau_var_14_7 ] = 0
 softmax_14_1: - exp_tau_var_14_1 + [ r[14,1] * exp_tau_var_14_0
   + r[14,1] * exp_tau_var_14_1 + r[14,1] * exp_tau_var_14_2
   + r[14,1] * exp_tau_var_14_3 + r[14,1] * exp_tau_var_14_4
   + r[14,1] * exp_tau_var_14_5 + r[14,1] * exp_tau_var_14_6
   + r[14,1] * exp_tau_var_14_7 ] = 0
 softmax_14_2: - exp_tau_var_14_2 + [ r[14,2] * exp_tau_var_14_0
   + r[14,2] * exp_tau_var_14_1 + r[14,2] * exp_tau_var_14_2
   + r[14,2] * exp_tau_var_14_3 + r[14,2] * exp_tau_var_14_4
   + r[14,2] * exp_tau_var_14_5 + r[14,2] * exp_tau_var_14_6
   + r[14,2] * exp_tau_var_14_7 ] = 0
 softmax_14_3: - exp_tau_var_14_3 + [ r[14,3] * exp_tau_var_14_0
   + r[14,3] * exp_tau_var_14_1 + r[14,3] * exp_tau_var_14_2
   + r[14,3] * exp_tau_var_14_3 + r[14,3] * exp_tau_var_14_4
   + r[14,3] * exp_tau_var_14_5 + r[14,3] * exp_tau_var_14_6
   + r[14,3] * exp_tau_var_14_7 ] = 0
 softmax_14_4: - exp_tau_var_14_4 + [ r[14,4] * exp_tau_var_14_0
   + r[14,4] * exp_tau_var_14_1 + r[14,4] * exp_tau_var_14_2
   + r[14,4] * exp_tau_var_14_3 + r[14,4] * exp_tau_var_14_4
   + r[14,4] * exp_tau_var_14_5 + r[14,4] * exp_tau_var_14_6
   + r[14,4] * exp_tau_var_14_7 ] = 0
 softmax_14_5: - exp_tau_var_14_5 + [ r[14,5] * exp_tau_var_14_0
   + r[14,5] * exp_tau_var_14_1 + r[14,5] * exp_tau_var_14_2
   + r[14,5] * exp_tau_var_14_3 + r[14,5] * exp_tau_var_14_4
   + r[14,5] * exp_tau_var_14_5 + r[14,5] * exp_tau_var_14_6
   + r[14,5] * exp_tau_var_14_7 ] = 0
 softmax_14_6: - exp_tau_var_14_6 + [ r[14,6] * exp_tau_var_14_0
   + r[14,6] * exp_tau_var_14_1 + r[14,6] * exp_tau_var_14_2
   + r[14,6] * exp_tau_var_14_3 + r[14,6] * exp_tau_var_14_4
   + r[14,6] * exp_tau_var_14_5 + r[14,6] * exp_tau_var_14_6
   + r[14,6] * exp_tau_var_14_7 ] = 0
 softmax_14_7: - exp_tau_var_14_7 + [ r[14,7] * exp_tau_var_14_0
   + r[14,7] * exp_tau_var_14_1 + r[14,7] * exp_tau_var_14_2
   + r[14,7] * exp_tau_var_14_3 + r[14,7] * exp_tau_var_14_4
   + r[14,7] * exp_tau_var_14_5 + r[14,7] * exp_tau_var_14_6
   + r[14,7] * exp_tau_var_14_7 ] = 0
 softmax_15_0: - exp_tau_var_15_0 + [ r[15,0] * exp_tau_var_15_0
   + r[15,0] * exp_tau_var_15_1 + r[15,0] * exp_tau_var_15_2
   + r[15,0] * exp_tau_var_15_3 + r[15,0] * exp_tau_var_15_4
   + r[15,0] * exp_tau_var_15_5 + r[15,0] * exp_tau_var_15_6
   + r[15,0] * exp_tau_var_15_7 ] = 0
 softmax_15_1: - exp_tau_var_15_1 + [ r[15,1] * exp_tau_var_15_0
   + r[15,1] * exp_tau_var_15_1 + r[15,1] * exp_tau_var_15_2
   + r[15,1] * exp_tau_var_15_3 + r[15,1] * exp_tau_var_15_4
   + r[15,1] * exp_tau_var_15_5 + r[15,1] * exp_tau_var_15_6
   + r[15,1] * exp_tau_var_15_7 ] = 0
 softmax_15_2: - exp_tau_var_15_2 + [ r[15,2] * exp_tau_var_15_0
   + r[15,2] * exp_tau_var_15_1 + r[15,2] * exp_tau_var_15_2
   + r[15,2] * exp_tau_var_15_3 + r[15,2] * exp_tau_var_15_4
   + r[15,2] * exp_tau_var_15_5 + r[15,2] * exp_tau_var_15_6
   + r[15,2] * exp_tau_var_15_7 ] = 0
 softmax_15_3: - exp_tau_var_15_3 + [ r[15,3] * exp_tau_var_15_0
   + r[15,3] * exp_tau_var_15_1 + r[15,3] * exp_tau_var_15_2
   + r[15,3] * exp_tau_var_15_3 + r[15,3] * exp_tau_var_15_4
   + r[15,3] * exp_tau_var_15_5 + r[15,3] * exp_tau_var_15_6
   + r[15,3] * exp_tau_var_15_7 ] = 0
 softmax_15_4: - exp_tau_var_15_4 + [ r[15,4] * exp_tau_var_15_0
   + r[15,4] * exp_tau_var_15_1 + r[15,4] * exp_tau_var_15_2
   + r[15,4] * exp_tau_var_15_3 + r[15,4] * exp_tau_var_15_4
   + r[15,4] * exp_tau_var_15_5 + r[15,4] * exp_tau_var_15_6
   + r[15,4] * exp_tau_var_15_7 ] = 0
 softmax_15_5: - exp_tau_var_15_5 + [ r[15,5] * exp_tau_var_15_0
   + r[15,5] * exp_tau_var_15_1 + r[15,5] * exp_tau_var_15_2
   + r[15,5] * exp_tau_var_15_3 + r[15,5] * exp_tau_var_15_4
   + r[15,5] * exp_tau_var_15_5 + r[15,5] * exp_tau_var_15_6
   + r[15,5] * exp_tau_var_15_7 ] = 0
 softmax_15_6: - exp_tau_var_15_6 + [ r[15,6] * exp_tau_var_15_0
   + r[15,6] * exp_tau_var_15_1 + r[15,6] * exp_tau_var_15_2
   + r[15,6] * exp_tau_var_15_3 + r[15,6] * exp_tau_var_15_4
   + r[15,6] * exp_tau_var_15_5 + r[15,6] * exp_tau_var_15_6
   + r[15,6] * exp_tau_var_15_7 ] = 0
 softmax_15_7: - exp_tau_var_15_7 + [ r[15,7] * exp_tau_var_15_0
   + r[15,7] * exp_tau_var_15_1 + r[15,7] * exp_tau_var_15_2
   + r[15,7] * exp_tau_var_15_3 + r[15,7] * exp_tau_var_15_4
   + r[15,7] * exp_tau_var_15_5 + r[15,7] * exp_tau_var_15_6
   + r[15,7] * exp_tau_var_15_7 ] = 0
 softmax_16_0: - exp_tau_var_16_0 + [ r[16,0] * exp_tau_var_16_0
   + r[16,0] * exp_tau_var_16_1 + r[16,0] * exp_tau_var_16_2
   + r[16,0] * exp_tau_var_16_3 + r[16,0] * exp_tau_var_16_4
   + r[16,0] * exp_tau_var_16_5 + r[16,0] * exp_tau_var_16_6
   + r[16,0] * exp_tau_var_16_7 ] = 0
 softmax_16_1: - exp_tau_var_16_1 + [ r[16,1] * exp_tau_var_16_0
   + r[16,1] * exp_tau_var_16_1 + r[16,1] * exp_tau_var_16_2
   + r[16,1] * exp_tau_var_16_3 + r[16,1] * exp_tau_var_16_4
   + r[16,1] * exp_tau_var_16_5 + r[16,1] * exp_tau_var_16_6
   + r[16,1] * exp_tau_var_16_7 ] = 0
 softmax_16_2: - exp_tau_var_16_2 + [ r[16,2] * exp_tau_var_16_0
   + r[16,2] * exp_tau_var_16_1 + r[16,2] * exp_tau_var_16_2
   + r[16,2] * exp_tau_var_16_3 + r[16,2] * exp_tau_var_16_4
   + r[16,2] * exp_tau_var_16_5 + r[16,2] * exp_tau_var_16_6
   + r[16,2] * exp_tau_var_16_7 ] = 0
 softmax_16_3: - exp_tau_var_16_3 + [ r[16,3] * exp_tau_var_16_0
   + r[16,3] * exp_tau_var_16_1 + r[16,3] * exp_tau_var_16_2
   + r[16,3] * exp_tau_var_16_3 + r[16,3] * exp_tau_var_16_4
   + r[16,3] * exp_tau_var_16_5 + r[16,3] * exp_tau_var_16_6
   + r[16,3] * exp_tau_var_16_7 ] = 0
 softmax_16_4: - exp_tau_var_16_4 + [ r[16,4] * exp_tau_var_16_0
   + r[16,4] * exp_tau_var_16_1 + r[16,4] * exp_tau_var_16_2
   + r[16,4] * exp_tau_var_16_3 + r[16,4] * exp_tau_var_16_4
   + r[16,4] * exp_tau_var_16_5 + r[16,4] * exp_tau_var_16_6
   + r[16,4] * exp_tau_var_16_7 ] = 0
 softmax_16_5: - exp_tau_var_16_5 + [ r[16,5] * exp_tau_var_16_0
   + r[16,5] * exp_tau_var_16_1 + r[16,5] * exp_tau_var_16_2
   + r[16,5] * exp_tau_var_16_3 + r[16,5] * exp_tau_var_16_4
   + r[16,5] * exp_tau_var_16_5 + r[16,5] * exp_tau_var_16_6
   + r[16,5] * exp_tau_var_16_7 ] = 0
 softmax_16_6: - exp_tau_var_16_6 + [ r[16,6] * exp_tau_var_16_0
   + r[16,6] * exp_tau_var_16_1 + r[16,6] * exp_tau_var_16_2
   + r[16,6] * exp_tau_var_16_3 + r[16,6] * exp_tau_var_16_4
   + r[16,6] * exp_tau_var_16_5 + r[16,6] * exp_tau_var_16_6
   + r[16,6] * exp_tau_var_16_7 ] = 0
 softmax_16_7: - exp_tau_var_16_7 + [ r[16,7] * exp_tau_var_16_0
   + r[16,7] * exp_tau_var_16_1 + r[16,7] * exp_tau_var_16_2
   + r[16,7] * exp_tau_var_16_3 + r[16,7] * exp_tau_var_16_4
   + r[16,7] * exp_tau_var_16_5 + r[16,7] * exp_tau_var_16_6
   + r[16,7] * exp_tau_var_16_7 ] = 0
 softmax_17_0: - exp_tau_var_17_0 + [ r[17,0] * exp_tau_var_17_0
   + r[17,0] * exp_tau_var_17_1 + r[17,0] * exp_tau_var_17_2
   + r[17,0] * exp_tau_var_17_3 + r[17,0] * exp_tau_var_17_4
   + r[17,0] * exp_tau_var_17_5 + r[17,0] * exp_tau_var_17_6
   + r[17,0] * exp_tau_var_17_7 ] = 0
 softmax_17_1: - exp_tau_var_17_1 + [ r[17,1] * exp_tau_var_17_0
   + r[17,1] * exp_tau_var_17_1 + r[17,1] * exp_tau_var_17_2
   + r[17,1] * exp_tau_var_17_3 + r[17,1] * exp_tau_var_17_4
   + r[17,1] * exp_tau_var_17_5 + r[17,1] * exp_tau_var_17_6
   + r[17,1] * exp_tau_var_17_7 ] = 0
 softmax_17_2: - exp_tau_var_17_2 + [ r[17,2] * exp_tau_var_17_0
   + r[17,2] * exp_tau_var_17_1 + r[17,2] * exp_tau_var_17_2
   + r[17,2] * exp_tau_var_17_3 + r[17,2] * exp_tau_var_17_4
   + r[17,2] * exp_tau_var_17_5 + r[17,2] * exp_tau_var_17_6
   + r[17,2] * exp_tau_var_17_7 ] = 0
 softmax_17_3: - exp_tau_var_17_3 + [ r[17,3] * exp_tau_var_17_0
   + r[17,3] * exp_tau_var_17_1 + r[17,3] * exp_tau_var_17_2
   + r[17,3] * exp_tau_var_17_3 + r[17,3] * exp_tau_var_17_4
   + r[17,3] * exp_tau_var_17_5 + r[17,3] * exp_tau_var_17_6
   + r[17,3] * exp_tau_var_17_7 ] = 0
 softmax_17_4: - exp_tau_var_17_4 + [ r[17,4] * exp_tau_var_17_0
   + r[17,4] * exp_tau_var_17_1 + r[17,4] * exp_tau_var_17_2
   + r[17,4] * exp_tau_var_17_3 + r[17,4] * exp_tau_var_17_4
   + r[17,4] * exp_tau_var_17_5 + r[17,4] * exp_tau_var_17_6
   + r[17,4] * exp_tau_var_17_7 ] = 0
 softmax_17_5: - exp_tau_var_17_5 + [ r[17,5] * exp_tau_var_17_0
   + r[17,5] * exp_tau_var_17_1 + r[17,5] * exp_tau_var_17_2
   + r[17,5] * exp_tau_var_17_3 + r[17,5] * exp_tau_var_17_4
   + r[17,5] * exp_tau_var_17_5 + r[17,5] * exp_tau_var_17_6
   + r[17,5] * exp_tau_var_17_7 ] = 0
 softmax_17_6: - exp_tau_var_17_6 + [ r[17,6] * exp_tau_var_17_0
   + r[17,6] * exp_tau_var_17_1 + r[17,6] * exp_tau_var_17_2
   + r[17,6] * exp_tau_var_17_3 + r[17,6] * exp_tau_var_17_4
   + r[17,6] * exp_tau_var_17_5 + r[17,6] * exp_tau_var_17_6
   + r[17,6] * exp_tau_var_17_7 ] = 0
 softmax_17_7: - exp_tau_var_17_7 + [ r[17,7] * exp_tau_var_17_0
   + r[17,7] * exp_tau_var_17_1 + r[17,7] * exp_tau_var_17_2
   + r[17,7] * exp_tau_var_17_3 + r[17,7] * exp_tau_var_17_4
   + r[17,7] * exp_tau_var_17_5 + r[17,7] * exp_tau_var_17_6
   + r[17,7] * exp_tau_var_17_7 ] = 0
 softmax_18_0: - exp_tau_var_18_0 + [ r[18,0] * exp_tau_var_18_0
   + r[18,0] * exp_tau_var_18_1 + r[18,0] * exp_tau_var_18_2
   + r[18,0] * exp_tau_var_18_3 + r[18,0] * exp_tau_var_18_4
   + r[18,0] * exp_tau_var_18_5 + r[18,0] * exp_tau_var_18_6
   + r[18,0] * exp_tau_var_18_7 ] = 0
 softmax_18_1: - exp_tau_var_18_1 + [ r[18,1] * exp_tau_var_18_0
   + r[18,1] * exp_tau_var_18_1 + r[18,1] * exp_tau_var_18_2
   + r[18,1] * exp_tau_var_18_3 + r[18,1] * exp_tau_var_18_4
   + r[18,1] * exp_tau_var_18_5 + r[18,1] * exp_tau_var_18_6
   + r[18,1] * exp_tau_var_18_7 ] = 0
 softmax_18_2: - exp_tau_var_18_2 + [ r[18,2] * exp_tau_var_18_0
   + r[18,2] * exp_tau_var_18_1 + r[18,2] * exp_tau_var_18_2
   + r[18,2] * exp_tau_var_18_3 + r[18,2] * exp_tau_var_18_4
   + r[18,2] * exp_tau_var_18_5 + r[18,2] * exp_tau_var_18_6
   + r[18,2] * exp_tau_var_18_7 ] = 0
 softmax_18_3: - exp_tau_var_18_3 + [ r[18,3] * exp_tau_var_18_0
   + r[18,3] * exp_tau_var_18_1 + r[18,3] * exp_tau_var_18_2
   + r[18,3] * exp_tau_var_18_3 + r[18,3] * exp_tau_var_18_4
   + r[18,3] * exp_tau_var_18_5 + r[18,3] * exp_tau_var_18_6
   + r[18,3] * exp_tau_var_18_7 ] = 0
 softmax_18_4: - exp_tau_var_18_4 + [ r[18,4] * exp_tau_var_18_0
   + r[18,4] * exp_tau_var_18_1 + r[18,4] * exp_tau_var_18_2
   + r[18,4] * exp_tau_var_18_3 + r[18,4] * exp_tau_var_18_4
   + r[18,4] * exp_tau_var_18_5 + r[18,4] * exp_tau_var_18_6
   + r[18,4] * exp_tau_var_18_7 ] = 0
 softmax_18_5: - exp_tau_var_18_5 + [ r[18,5] * exp_tau_var_18_0
   + r[18,5] * exp_tau_var_18_1 + r[18,5] * exp_tau_var_18_2
   + r[18,5] * exp_tau_var_18_3 + r[18,5] * exp_tau_var_18_4
   + r[18,5] * exp_tau_var_18_5 + r[18,5] * exp_tau_var_18_6
   + r[18,5] * exp_tau_var_18_7 ] = 0
 softmax_18_6: - exp_tau_var_18_6 + [ r[18,6] * exp_tau_var_18_0
   + r[18,6] * exp_tau_var_18_1 + r[18,6] * exp_tau_var_18_2
   + r[18,6] * exp_tau_var_18_3 + r[18,6] * exp_tau_var_18_4
   + r[18,6] * exp_tau_var_18_5 + r[18,6] * exp_tau_var_18_6
   + r[18,6] * exp_tau_var_18_7 ] = 0
 softmax_18_7: - exp_tau_var_18_7 + [ r[18,7] * exp_tau_var_18_0
   + r[18,7] * exp_tau_var_18_1 + r[18,7] * exp_tau_var_18_2
   + r[18,7] * exp_tau_var_18_3 + r[18,7] * exp_tau_var_18_4
   + r[18,7] * exp_tau_var_18_5 + r[18,7] * exp_tau_var_18_6
   + r[18,7] * exp_tau_var_18_7 ] = 0
 softmax_19_0: - exp_tau_var_19_0 + [ r[19,0] * exp_tau_var_19_0
   + r[19,0] * exp_tau_var_19_1 + r[19,0] * exp_tau_var_19_2
   + r[19,0] * exp_tau_var_19_3 + r[19,0] * exp_tau_var_19_4
   + r[19,0] * exp_tau_var_19_5 + r[19,0] * exp_tau_var_19_6
   + r[19,0] * exp_tau_var_19_7 ] = 0
 softmax_19_1: - exp_tau_var_19_1 + [ r[19,1] * exp_tau_var_19_0
   + r[19,1] * exp_tau_var_19_1 + r[19,1] * exp_tau_var_19_2
   + r[19,1] * exp_tau_var_19_3 + r[19,1] * exp_tau_var_19_4
   + r[19,1] * exp_tau_var_19_5 + r[19,1] * exp_tau_var_19_6
   + r[19,1] * exp_tau_var_19_7 ] = 0
 softmax_19_2: - exp_tau_var_19_2 + [ r[19,2] * exp_tau_var_19_0
   + r[19,2] * exp_tau_var_19_1 + r[19,2] * exp_tau_var_19_2
   + r[19,2] * exp_tau_var_19_3 + r[19,2] * exp_tau_var_19_4
   + r[19,2] * exp_tau_var_19_5 + r[19,2] * exp_tau_var_19_6
   + r[19,2] * exp_tau_var_19_7 ] = 0
 softmax_19_3: - exp_tau_var_19_3 + [ r[19,3] * exp_tau_var_19_0
   + r[19,3] * exp_tau_var_19_1 + r[19,3] * exp_tau_var_19_2
   + r[19,3] * exp_tau_var_19_3 + r[19,3] * exp_tau_var_19_4
   + r[19,3] * exp_tau_var_19_5 + r[19,3] * exp_tau_var_19_6
   + r[19,3] * exp_tau_var_19_7 ] = 0
 softmax_19_4: - exp_tau_var_19_4 + [ r[19,4] * exp_tau_var_19_0
   + r[19,4] * exp_tau_var_19_1 + r[19,4] * exp_tau_var_19_2
   + r[19,4] * exp_tau_var_19_3 + r[19,4] * exp_tau_var_19_4
   + r[19,4] * exp_tau_var_19_5 + r[19,4] * exp_tau_var_19_6
   + r[19,4] * exp_tau_var_19_7 ] = 0
 softmax_19_5: - exp_tau_var_19_5 + [ r[19,5] * exp_tau_var_19_0
   + r[19,5] * exp_tau_var_19_1 + r[19,5] * exp_tau_var_19_2
   + r[19,5] * exp_tau_var_19_3 + r[19,5] * exp_tau_var_19_4
   + r[19,5] * exp_tau_var_19_5 + r[19,5] * exp_tau_var_19_6
   + r[19,5] * exp_tau_var_19_7 ] = 0
 softmax_19_6: - exp_tau_var_19_6 + [ r[19,6] * exp_tau_var_19_0
   + r[19,6] * exp_tau_var_19_1 + r[19,6] * exp_tau_var_19_2
   + r[19,6] * exp_tau_var_19_3 + r[19,6] * exp_tau_var_19_4
   + r[19,6] * exp_tau_var_19_5 + r[19,6] * exp_tau_var_19_6
   + r[19,6] * exp_tau_var_19_7 ] = 0
 softmax_19_7: - exp_tau_var_19_7 + [ r[19,7] * exp_tau_var_19_0
   + r[19,7] * exp_tau_var_19_1 + r[19,7] * exp_tau_var_19_2
   + r[19,7] * exp_tau_var_19_3 + r[19,7] * exp_tau_var_19_4
   + r[19,7] * exp_tau_var_19_5 + r[19,7] * exp_tau_var_19_6
   + r[19,7] * exp_tau_var_19_7 ] = 0
 softmax_20_0: - exp_tau_var_20_0 + [ r[20,0] * exp_tau_var_20_0
   + r[20,0] * exp_tau_var_20_1 + r[20,0] * exp_tau_var_20_2
   + r[20,0] * exp_tau_var_20_3 + r[20,0] * exp_tau_var_20_4
   + r[20,0] * exp_tau_var_20_5 + r[20,0] * exp_tau_var_20_6
   + r[20,0] * exp_tau_var_20_7 ] = 0
 softmax_20_1: - exp_tau_var_20_1 + [ r[20,1] * exp_tau_var_20_0
   + r[20,1] * exp_tau_var_20_1 + r[20,1] * exp_tau_var_20_2
   + r[20,1] * exp_tau_var_20_3 + r[20,1] * exp_tau_var_20_4
   + r[20,1] * exp_tau_var_20_5 + r[20,1] * exp_tau_var_20_6
   + r[20,1] * exp_tau_var_20_7 ] = 0
 softmax_20_2: - exp_tau_var_20_2 + [ r[20,2] * exp_tau_var_20_0
   + r[20,2] * exp_tau_var_20_1 + r[20,2] * exp_tau_var_20_2
   + r[20,2] * exp_tau_var_20_3 + r[20,2] * exp_tau_var_20_4
   + r[20,2] * exp_tau_var_20_5 + r[20,2] * exp_tau_var_20_6
   + r[20,2] * exp_tau_var_20_7 ] = 0
 softmax_20_3: - exp_tau_var_20_3 + [ r[20,3] * exp_tau_var_20_0
   + r[20,3] * exp_tau_var_20_1 + r[20,3] * exp_tau_var_20_2
   + r[20,3] * exp_tau_var_20_3 + r[20,3] * exp_tau_var_20_4
   + r[20,3] * exp_tau_var_20_5 + r[20,3] * exp_tau_var_20_6
   + r[20,3] * exp_tau_var_20_7 ] = 0
 softmax_20_4: - exp_tau_var_20_4 + [ r[20,4] * exp_tau_var_20_0
   + r[20,4] * exp_tau_var_20_1 + r[20,4] * exp_tau_var_20_2
   + r[20,4] * exp_tau_var_20_3 + r[20,4] * exp_tau_var_20_4
   + r[20,4] * exp_tau_var_20_5 + r[20,4] * exp_tau_var_20_6
   + r[20,4] * exp_tau_var_20_7 ] = 0
 softmax_20_5: - exp_tau_var_20_5 + [ r[20,5] * exp_tau_var_20_0
   + r[20,5] * exp_tau_var_20_1 + r[20,5] * exp_tau_var_20_2
   + r[20,5] * exp_tau_var_20_3 + r[20,5] * exp_tau_var_20_4
   + r[20,5] * exp_tau_var_20_5 + r[20,5] * exp_tau_var_20_6
   + r[20,5] * exp_tau_var_20_7 ] = 0
 softmax_20_6: - exp_tau_var_20_6 + [ r[20,6] * exp_tau_var_20_0
   + r[20,6] * exp_tau_var_20_1 + r[20,6] * exp_tau_var_20_2
   + r[20,6] * exp_tau_var_20_3 + r[20,6] * exp_tau_var_20_4
   + r[20,6] * exp_tau_var_20_5 + r[20,6] * exp_tau_var_20_6
   + r[20,6] * exp_tau_var_20_7 ] = 0
 softmax_20_7: - exp_tau_var_20_7 + [ r[20,7] * exp_tau_var_20_0
   + r[20,7] * exp_tau_var_20_1 + r[20,7] * exp_tau_var_20_2
   + r[20,7] * exp_tau_var_20_3 + r[20,7] * exp_tau_var_20_4
   + r[20,7] * exp_tau_var_20_5 + r[20,7] * exp_tau_var_20_6
   + r[20,7] * exp_tau_var_20_7 ] = 0
 softmax_21_0: - exp_tau_var_21_0 + [ r[21,0] * exp_tau_var_21_0
   + r[21,0] * exp_tau_var_21_1 + r[21,0] * exp_tau_var_21_2
   + r[21,0] * exp_tau_var_21_3 + r[21,0] * exp_tau_var_21_4
   + r[21,0] * exp_tau_var_21_5 + r[21,0] * exp_tau_var_21_6
   + r[21,0] * exp_tau_var_21_7 ] = 0
 softmax_21_1: - exp_tau_var_21_1 + [ r[21,1] * exp_tau_var_21_0
   + r[21,1] * exp_tau_var_21_1 + r[21,1] * exp_tau_var_21_2
   + r[21,1] * exp_tau_var_21_3 + r[21,1] * exp_tau_var_21_4
   + r[21,1] * exp_tau_var_21_5 + r[21,1] * exp_tau_var_21_6
   + r[21,1] * exp_tau_var_21_7 ] = 0
 softmax_21_2: - exp_tau_var_21_2 + [ r[21,2] * exp_tau_var_21_0
   + r[21,2] * exp_tau_var_21_1 + r[21,2] * exp_tau_var_21_2
   + r[21,2] * exp_tau_var_21_3 + r[21,2] * exp_tau_var_21_4
   + r[21,2] * exp_tau_var_21_5 + r[21,2] * exp_tau_var_21_6
   + r[21,2] * exp_tau_var_21_7 ] = 0
 softmax_21_3: - exp_tau_var_21_3 + [ r[21,3] * exp_tau_var_21_0
   + r[21,3] * exp_tau_var_21_1 + r[21,3] * exp_tau_var_21_2
   + r[21,3] * exp_tau_var_21_3 + r[21,3] * exp_tau_var_21_4
   + r[21,3] * exp_tau_var_21_5 + r[21,3] * exp_tau_var_21_6
   + r[21,3] * exp_tau_var_21_7 ] = 0
 softmax_21_4: - exp_tau_var_21_4 + [ r[21,4] * exp_tau_var_21_0
   + r[21,4] * exp_tau_var_21_1 + r[21,4] * exp_tau_var_21_2
   + r[21,4] * exp_tau_var_21_3 + r[21,4] * exp_tau_var_21_4
   + r[21,4] * exp_tau_var_21_5 + r[21,4] * exp_tau_var_21_6
   + r[21,4] * exp_tau_var_21_7 ] = 0
 softmax_21_5: - exp_tau_var_21_5 + [ r[21,5] * exp_tau_var_21_0
   + r[21,5] * exp_tau_var_21_1 + r[21,5] * exp_tau_var_21_2
   + r[21,5] * exp_tau_var_21_3 + r[21,5] * exp_tau_var_21_4
   + r[21,5] * exp_tau_var_21_5 + r[21,5] * exp_tau_var_21_6
   + r[21,5] * exp_tau_var_21_7 ] = 0
 softmax_21_6: - exp_tau_var_21_6 + [ r[21,6] * exp_tau_var_21_0
   + r[21,6] * exp_tau_var_21_1 + r[21,6] * exp_tau_var_21_2
   + r[21,6] * exp_tau_var_21_3 + r[21,6] * exp_tau_var_21_4
   + r[21,6] * exp_tau_var_21_5 + r[21,6] * exp_tau_var_21_6
   + r[21,6] * exp_tau_var_21_7 ] = 0
 softmax_21_7: - exp_tau_var_21_7 + [ r[21,7] * exp_tau_var_21_0
   + r[21,7] * exp_tau_var_21_1 + r[21,7] * exp_tau_var_21_2
   + r[21,7] * exp_tau_var_21_3 + r[21,7] * exp_tau_var_21_4
   + r[21,7] * exp_tau_var_21_5 + r[21,7] * exp_tau_var_21_6
   + r[21,7] * exp_tau_var_21_7 ] = 0
 softmax_22_0: - exp_tau_var_22_0 + [ r[22,0] * exp_tau_var_22_0
   + r[22,0] * exp_tau_var_22_1 + r[22,0] * exp_tau_var_22_2
   + r[22,0] * exp_tau_var_22_3 + r[22,0] * exp_tau_var_22_4
   + r[22,0] * exp_tau_var_22_5 + r[22,0] * exp_tau_var_22_6
   + r[22,0] * exp_tau_var_22_7 ] = 0
 softmax_22_1: - exp_tau_var_22_1 + [ r[22,1] * exp_tau_var_22_0
   + r[22,1] * exp_tau_var_22_1 + r[22,1] * exp_tau_var_22_2
   + r[22,1] * exp_tau_var_22_3 + r[22,1] * exp_tau_var_22_4
   + r[22,1] * exp_tau_var_22_5 + r[22,1] * exp_tau_var_22_6
   + r[22,1] * exp_tau_var_22_7 ] = 0
 softmax_22_2: - exp_tau_var_22_2 + [ r[22,2] * exp_tau_var_22_0
   + r[22,2] * exp_tau_var_22_1 + r[22,2] * exp_tau_var_22_2
   + r[22,2] * exp_tau_var_22_3 + r[22,2] * exp_tau_var_22_4
   + r[22,2] * exp_tau_var_22_5 + r[22,2] * exp_tau_var_22_6
   + r[22,2] * exp_tau_var_22_7 ] = 0
 softmax_22_3: - exp_tau_var_22_3 + [ r[22,3] * exp_tau_var_22_0
   + r[22,3] * exp_tau_var_22_1 + r[22,3] * exp_tau_var_22_2
   + r[22,3] * exp_tau_var_22_3 + r[22,3] * exp_tau_var_22_4
   + r[22,3] * exp_tau_var_22_5 + r[22,3] * exp_tau_var_22_6
   + r[22,3] * exp_tau_var_22_7 ] = 0
 softmax_22_4: - exp_tau_var_22_4 + [ r[22,4] * exp_tau_var_22_0
   + r[22,4] * exp_tau_var_22_1 + r[22,4] * exp_tau_var_22_2
   + r[22,4] * exp_tau_var_22_3 + r[22,4] * exp_tau_var_22_4
   + r[22,4] * exp_tau_var_22_5 + r[22,4] * exp_tau_var_22_6
   + r[22,4] * exp_tau_var_22_7 ] = 0
 softmax_22_5: - exp_tau_var_22_5 + [ r[22,5] * exp_tau_var_22_0
   + r[22,5] * exp_tau_var_22_1 + r[22,5] * exp_tau_var_22_2
   + r[22,5] * exp_tau_var_22_3 + r[22,5] * exp_tau_var_22_4
   + r[22,5] * exp_tau_var_22_5 + r[22,5] * exp_tau_var_22_6
   + r[22,5] * exp_tau_var_22_7 ] = 0
 softmax_22_6: - exp_tau_var_22_6 + [ r[22,6] * exp_tau_var_22_0
   + r[22,6] * exp_tau_var_22_1 + r[22,6] * exp_tau_var_22_2
   + r[22,6] * exp_tau_var_22_3 + r[22,6] * exp_tau_var_22_4
   + r[22,6] * exp_tau_var_22_5 + r[22,6] * exp_tau_var_22_6
   + r[22,6] * exp_tau_var_22_7 ] = 0
 softmax_22_7: - exp_tau_var_22_7 + [ r[22,7] * exp_tau_var_22_0
   + r[22,7] * exp_tau_var_22_1 + r[22,7] * exp_tau_var_22_2
   + r[22,7] * exp_tau_var_22_3 + r[22,7] * exp_tau_var_22_4
   + r[22,7] * exp_tau_var_22_5 + r[22,7] * exp_tau_var_22_6
   + r[22,7] * exp_tau_var_22_7 ] = 0
 softmax_23_0: - exp_tau_var_23_0 + [ r[23,0] * exp_tau_var_23_0
   + r[23,0] * exp_tau_var_23_1 + r[23,0] * exp_tau_var_23_2
   + r[23,0] * exp_tau_var_23_3 + r[23,0] * exp_tau_var_23_4
   + r[23,0] * exp_tau_var_23_5 + r[23,0] * exp_tau_var_23_6
   + r[23,0] * exp_tau_var_23_7 ] = 0
 softmax_23_1: - exp_tau_var_23_1 + [ r[23,1] * exp_tau_var_23_0
   + r[23,1] * exp_tau_var_23_1 + r[23,1] * exp_tau_var_23_2
   + r[23,1] * exp_tau_var_23_3 + r[23,1] * exp_tau_var_23_4
   + r[23,1] * exp_tau_var_23_5 + r[23,1] * exp_tau_var_23_6
   + r[23,1] * exp_tau_var_23_7 ] = 0
 softmax_23_2: - exp_tau_var_23_2 + [ r[23,2] * exp_tau_var_23_0
   + r[23,2] * exp_tau_var_23_1 + r[23,2] * exp_tau_var_23_2
   + r[23,2] * exp_tau_var_23_3 + r[23,2] * exp_tau_var_23_4
   + r[23,2] * exp_tau_var_23_5 + r[23,2] * exp_tau_var_23_6
   + r[23,2] * exp_tau_var_23_7 ] = 0
 softmax_23_3: - exp_tau_var_23_3 + [ r[23,3] * exp_tau_var_23_0
   + r[23,3] * exp_tau_var_23_1 + r[23,3] * exp_tau_var_23_2
   + r[23,3] * exp_tau_var_23_3 + r[23,3] * exp_tau_var_23_4
   + r[23,3] * exp_tau_var_23_5 + r[23,3] * exp_tau_var_23_6
   + r[23,3] * exp_tau_var_23_7 ] = 0
 softmax_23_4: - exp_tau_var_23_4 + [ r[23,4] * exp_tau_var_23_0
   + r[23,4] * exp_tau_var_23_1 + r[23,4] * exp_tau_var_23_2
   + r[23,4] * exp_tau_var_23_3 + r[23,4] * exp_tau_var_23_4
   + r[23,4] * exp_tau_var_23_5 + r[23,4] * exp_tau_var_23_6
   + r[23,4] * exp_tau_var_23_7 ] = 0
 softmax_23_5: - exp_tau_var_23_5 + [ r[23,5] * exp_tau_var_23_0
   + r[23,5] * exp_tau_var_23_1 + r[23,5] * exp_tau_var_23_2
   + r[23,5] * exp_tau_var_23_3 + r[23,5] * exp_tau_var_23_4
   + r[23,5] * exp_tau_var_23_5 + r[23,5] * exp_tau_var_23_6
   + r[23,5] * exp_tau_var_23_7 ] = 0
 softmax_23_6: - exp_tau_var_23_6 + [ r[23,6] * exp_tau_var_23_0
   + r[23,6] * exp_tau_var_23_1 + r[23,6] * exp_tau_var_23_2
   + r[23,6] * exp_tau_var_23_3 + r[23,6] * exp_tau_var_23_4
   + r[23,6] * exp_tau_var_23_5 + r[23,6] * exp_tau_var_23_6
   + r[23,6] * exp_tau_var_23_7 ] = 0
 softmax_23_7: - exp_tau_var_23_7 + [ r[23,7] * exp_tau_var_23_0
   + r[23,7] * exp_tau_var_23_1 + r[23,7] * exp_tau_var_23_2
   + r[23,7] * exp_tau_var_23_3 + r[23,7] * exp_tau_var_23_4
   + r[23,7] * exp_tau_var_23_5 + r[23,7] * exp_tau_var_23_6
   + r[23,7] * exp_tau_var_23_7 ] = 0
 softmax_24_0: - exp_tau_var_24_0 + [ r[24,0] * exp_tau_var_24_0
   + r[24,0] * exp_tau_var_24_1 + r[24,0] * exp_tau_var_24_2
   + r[24,0] * exp_tau_var_24_3 + r[24,0] * exp_tau_var_24_4
   + r[24,0] * exp_tau_var_24_5 + r[24,0] * exp_tau_var_24_6
   + r[24,0] * exp_tau_var_24_7 ] = 0
 softmax_24_1: - exp_tau_var_24_1 + [ r[24,1] * exp_tau_var_24_0
   + r[24,1] * exp_tau_var_24_1 + r[24,1] * exp_tau_var_24_2
   + r[24,1] * exp_tau_var_24_3 + r[24,1] * exp_tau_var_24_4
   + r[24,1] * exp_tau_var_24_5 + r[24,1] * exp_tau_var_24_6
   + r[24,1] * exp_tau_var_24_7 ] = 0
 softmax_24_2: - exp_tau_var_24_2 + [ r[24,2] * exp_tau_var_24_0
   + r[24,2] * exp_tau_var_24_1 + r[24,2] * exp_tau_var_24_2
   + r[24,2] * exp_tau_var_24_3 + r[24,2] * exp_tau_var_24_4
   + r[24,2] * exp_tau_var_24_5 + r[24,2] * exp_tau_var_24_6
   + r[24,2] * exp_tau_var_24_7 ] = 0
 softmax_24_3: - exp_tau_var_24_3 + [ r[24,3] * exp_tau_var_24_0
   + r[24,3] * exp_tau_var_24_1 + r[24,3] * exp_tau_var_24_2
   + r[24,3] * exp_tau_var_24_3 + r[24,3] * exp_tau_var_24_4
   + r[24,3] * exp_tau_var_24_5 + r[24,3] * exp_tau_var_24_6
   + r[24,3] * exp_tau_var_24_7 ] = 0
 softmax_24_4: - exp_tau_var_24_4 + [ r[24,4] * exp_tau_var_24_0
   + r[24,4] * exp_tau_var_24_1 + r[24,4] * exp_tau_var_24_2
   + r[24,4] * exp_tau_var_24_3 + r[24,4] * exp_tau_var_24_4
   + r[24,4] * exp_tau_var_24_5 + r[24,4] * exp_tau_var_24_6
   + r[24,4] * exp_tau_var_24_7 ] = 0
 softmax_24_5: - exp_tau_var_24_5 + [ r[24,5] * exp_tau_var_24_0
   + r[24,5] * exp_tau_var_24_1 + r[24,5] * exp_tau_var_24_2
   + r[24,5] * exp_tau_var_24_3 + r[24,5] * exp_tau_var_24_4
   + r[24,5] * exp_tau_var_24_5 + r[24,5] * exp_tau_var_24_6
   + r[24,5] * exp_tau_var_24_7 ] = 0
 softmax_24_6: - exp_tau_var_24_6 + [ r[24,6] * exp_tau_var_24_0
   + r[24,6] * exp_tau_var_24_1 + r[24,6] * exp_tau_var_24_2
   + r[24,6] * exp_tau_var_24_3 + r[24,6] * exp_tau_var_24_4
   + r[24,6] * exp_tau_var_24_5 + r[24,6] * exp_tau_var_24_6
   + r[24,6] * exp_tau_var_24_7 ] = 0
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
 betas[0] free
 betas[1] free
 betas[2] free
 betas[3] free
 betas[4] free
 betas[5] free
 betas[6] free
 betas[7] free
 gammas[0] free
 gammas[1] free
 gammas[2] free
 gammas[3] free
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
 Q0_var[0] <= 2054.400566507637
 Q0_var[1] <= 2054.400566507637
 Q0_var[2] <= 2054.400566507637
 Q0_var[3] <= 2054.400566507637
 Q0_var[4] <= 2054.400566507637
 Q0_var[5] <= 2054.400566507637
 Q0_var[6] <= 2054.400566507637
 Q0_var[7] <= 2054.400566507637
 Q0_var[8] <= 2054.400566507637
 Q0_var[9] <= 2054.400566507637
 Q0_var[10] <= 2054.400566507637
 Q0_var[11] <= 2054.400566507637
 Q0_var[12] <= 2054.400566507637
 Q0_var[13] <= 2054.400566507637
 Q0_var[14] <= 2054.400566507637
 Q0_var[15] <= 2054.400566507637
 Q0_var[16] <= 2054.400566507637
 Q0_var[17] <= 2054.400566507637
 Q0_var[18] <= 2054.400566507637
 Q0_var[19] <= 2054.400566507637
 Q0_var[20] <= 2054.400566507637
 Q0_var[21] <= 2054.400566507637
 Q0_var[22] <= 2054.400566507637
 Q0_var[23] <= 2054.400566507637
 Q0_var[24] <= 2054.400566507637
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
 max_r_helper <= 1
 Left_0_aux_10 free
 Lost_0_aux_10 free
 Left_0_aux_10 free
 Lost_0_aux_10 free
 max_r_helper <= 1
 Left_0_aux_11 free
 Lost_0_aux_11 free
 Left_0_aux_11 free
 Lost_0_aux_11 free
 max_r_helper <= 1
 Left_0_aux_12 free
 Lost_0_aux_12 free
 Left_0_aux_12 free
 Lost_0_aux_12 free
 max_r_helper <= 1
 Left_0_aux_13 free
 Lost_0_aux_13 free
 Left_0_aux_13 free
 Lost_0_aux_13 free
 max_r_helper <= 1
 Left_0_aux_14 free
 Lost_0_aux_14 free
 Left_0_aux_14 free
 Lost_0_aux_14 free
 max_r_helper <= 1
 Left_0_aux_15 free
 Lost_0_aux_15 free
 Left_0_aux_15 free
 Lost_0_aux_15 free
 max_r_helper <= 1
 Left_0_aux_16 free
 Lost_0_aux_16 free
 Left_0_aux_16 free
 Lost_0_aux_16 free
 max_r_helper <= 1
 Left_0_aux_17 free
 Lost_0_aux_17 free
 Left_0_aux_17 free
 Lost_0_aux_17 free
 max_r_helper <= 1
 Left_0_aux_18 free
 Lost_0_aux_18 free
 Left_0_aux_18 free
 Lost_0_aux_18 free
 max_r_helper <= 1
 Left_0_aux_19 free
 Lost_0_aux_19 free
 Left_0_aux_19 free
 Lost_0_aux_19 free
 max_r_helper <= 1
 Left_0_aux_20 free
 Lost_0_aux_20 free
 Left_0_aux_20 free
 Lost_0_aux_20 free
 max_r_helper <= 1
 Left_0_aux_21 free
 Lost_0_aux_21 free
 Left_0_aux_21 free
 Lost_0_aux_21 free
 max_r_helper <= 1
 Left_0_aux_22 free
 Lost_0_aux_22 free
 Left_0_aux_22 free
 Lost_0_aux_22 free
 max_r_helper <= 1
 Left_0_aux_23 free
 Lost_0_aux_23 free
 Left_0_aux_23 free
 Lost_0_aux_23 free
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
 logistic_constraint_10: Fraction_for_second_order_amount[10] =
   LOGISTIC ( f_var[10] )
 GC251: exp_tau_var_10_0 = EXP ( tau[10,0] )
 GC252: exp_tau_var_10_1 = EXP ( tau[10,1] )
 GC253: exp_tau_var_10_2 = EXP ( tau[10,2] )
 GC254: exp_tau_var_10_3 = EXP ( tau[10,3] )
 GC255: exp_tau_var_10_4 = EXP ( tau[10,4] )
 GC256: exp_tau_var_10_5 = EXP ( tau[10,5] )
 GC257: exp_tau_var_10_6 = EXP ( tau[10,6] )
 GC258: exp_tau_var_10_7 = EXP ( tau[10,7] )
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
 GC276: exp_tau_var_11_0 = EXP ( tau[11,0] )
 GC277: exp_tau_var_11_1 = EXP ( tau[11,1] )
 GC278: exp_tau_var_11_2 = EXP ( tau[11,2] )
 GC279: exp_tau_var_11_3 = EXP ( tau[11,3] )
 GC280: exp_tau_var_11_4 = EXP ( tau[11,4] )
 GC281: exp_tau_var_11_5 = EXP ( tau[11,5] )
 GC282: exp_tau_var_11_6 = EXP ( tau[11,6] )
 GC283: exp_tau_var_11_7 = EXP ( tau[11,7] )
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
 GC301: exp_tau_var_12_0 = EXP ( tau[12,0] )
 GC302: exp_tau_var_12_1 = EXP ( tau[12,1] )
 GC303: exp_tau_var_12_2 = EXP ( tau[12,2] )
 GC304: exp_tau_var_12_3 = EXP ( tau[12,3] )
 GC305: exp_tau_var_12_4 = EXP ( tau[12,4] )
 GC306: exp_tau_var_12_5 = EXP ( tau[12,5] )
 GC307: exp_tau_var_12_6 = EXP ( tau[12,6] )
 GC308: exp_tau_var_12_7 = EXP ( tau[12,7] )
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
 GC326: exp_tau_var_13_0 = EXP ( tau[13,0] )
 GC327: exp_tau_var_13_1 = EXP ( tau[13,1] )
 GC328: exp_tau_var_13_2 = EXP ( tau[13,2] )
 GC329: exp_tau_var_13_3 = EXP ( tau[13,3] )
 GC330: exp_tau_var_13_4 = EXP ( tau[13,4] )
 GC331: exp_tau_var_13_5 = EXP ( tau[13,5] )
 GC332: exp_tau_var_13_6 = EXP ( tau[13,6] )
 GC333: exp_tau_var_13_7 = EXP ( tau[13,7] )
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
 GC351: exp_tau_var_14_0 = EXP ( tau[14,0] )
 GC352: exp_tau_var_14_1 = EXP ( tau[14,1] )
 GC353: exp_tau_var_14_2 = EXP ( tau[14,2] )
 GC354: exp_tau_var_14_3 = EXP ( tau[14,3] )
 GC355: exp_tau_var_14_4 = EXP ( tau[14,4] )
 GC356: exp_tau_var_14_5 = EXP ( tau[14,5] )
 GC357: exp_tau_var_14_6 = EXP ( tau[14,6] )
 GC358: exp_tau_var_14_7 = EXP ( tau[14,7] )
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
 GC376: exp_tau_var_15_0 = EXP ( tau[15,0] )
 GC377: exp_tau_var_15_1 = EXP ( tau[15,1] )
 GC378: exp_tau_var_15_2 = EXP ( tau[15,2] )
 GC379: exp_tau_var_15_3 = EXP ( tau[15,3] )
 GC380: exp_tau_var_15_4 = EXP ( tau[15,4] )
 GC381: exp_tau_var_15_5 = EXP ( tau[15,5] )
 GC382: exp_tau_var_15_6 = EXP ( tau[15,6] )
 GC383: exp_tau_var_15_7 = EXP ( tau[15,7] )
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
 GC401: exp_tau_var_16_0 = EXP ( tau[16,0] )
 GC402: exp_tau_var_16_1 = EXP ( tau[16,1] )
 GC403: exp_tau_var_16_2 = EXP ( tau[16,2] )
 GC404: exp_tau_var_16_3 = EXP ( tau[16,3] )
 GC405: exp_tau_var_16_4 = EXP ( tau[16,4] )
 GC406: exp_tau_var_16_5 = EXP ( tau[16,5] )
 GC407: exp_tau_var_16_6 = EXP ( tau[16,6] )
 GC408: exp_tau_var_16_7 = EXP ( tau[16,7] )
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
 GC426: exp_tau_var_17_0 = EXP ( tau[17,0] )
 GC427: exp_tau_var_17_1 = EXP ( tau[17,1] )
 GC428: exp_tau_var_17_2 = EXP ( tau[17,2] )
 GC429: exp_tau_var_17_3 = EXP ( tau[17,3] )
 GC430: exp_tau_var_17_4 = EXP ( tau[17,4] )
 GC431: exp_tau_var_17_5 = EXP ( tau[17,5] )
 GC432: exp_tau_var_17_6 = EXP ( tau[17,6] )
 GC433: exp_tau_var_17_7 = EXP ( tau[17,7] )
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
 GC451: exp_tau_var_18_0 = EXP ( tau[18,0] )
 GC452: exp_tau_var_18_1 = EXP ( tau[18,1] )
 GC453: exp_tau_var_18_2 = EXP ( tau[18,2] )
 GC454: exp_tau_var_18_3 = EXP ( tau[18,3] )
 GC455: exp_tau_var_18_4 = EXP ( tau[18,4] )
 GC456: exp_tau_var_18_5 = EXP ( tau[18,5] )
 GC457: exp_tau_var_18_6 = EXP ( tau[18,6] )
 GC458: exp_tau_var_18_7 = EXP ( tau[18,7] )
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
 GC476: exp_tau_var_19_0 = EXP ( tau[19,0] )
 GC477: exp_tau_var_19_1 = EXP ( tau[19,1] )
 GC478: exp_tau_var_19_2 = EXP ( tau[19,2] )
 GC479: exp_tau_var_19_3 = EXP ( tau[19,3] )
 GC480: exp_tau_var_19_4 = EXP ( tau[19,4] )
 GC481: exp_tau_var_19_5 = EXP ( tau[19,5] )
 GC482: exp_tau_var_19_6 = EXP ( tau[19,6] )
 GC483: exp_tau_var_19_7 = EXP ( tau[19,7] )
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
 GC501: exp_tau_var_20_0 = EXP ( tau[20,0] )
 GC502: exp_tau_var_20_1 = EXP ( tau[20,1] )
 GC503: exp_tau_var_20_2 = EXP ( tau[20,2] )
 GC504: exp_tau_var_20_3 = EXP ( tau[20,3] )
 GC505: exp_tau_var_20_4 = EXP ( tau[20,4] )
 GC506: exp_tau_var_20_5 = EXP ( tau[20,5] )
 GC507: exp_tau_var_20_6 = EXP ( tau[20,6] )
 GC508: exp_tau_var_20_7 = EXP ( tau[20,7] )
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
 GC526: exp_tau_var_21_0 = EXP ( tau[21,0] )
 GC527: exp_tau_var_21_1 = EXP ( tau[21,1] )
 GC528: exp_tau_var_21_2 = EXP ( tau[21,2] )
 GC529: exp_tau_var_21_3 = EXP ( tau[21,3] )
 GC530: exp_tau_var_21_4 = EXP ( tau[21,4] )
 GC531: exp_tau_var_21_5 = EXP ( tau[21,5] )
 GC532: exp_tau_var_21_6 = EXP ( tau[21,6] )
 GC533: exp_tau_var_21_7 = EXP ( tau[21,7] )
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
 GC551: exp_tau_var_22_0 = EXP ( tau[22,0] )
 GC552: exp_tau_var_22_1 = EXP ( tau[22,1] )
 GC553: exp_tau_var_22_2 = EXP ( tau[22,2] )
 GC554: exp_tau_var_22_3 = EXP ( tau[22,3] )
 GC555: exp_tau_var_22_4 = EXP ( tau[22,4] )
 GC556: exp_tau_var_22_5 = EXP ( tau[22,5] )
 GC557: exp_tau_var_22_6 = EXP ( tau[22,6] )
 GC558: exp_tau_var_22_7 = EXP ( tau[22,7] )
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
 GC576: exp_tau_var_23_0 = EXP ( tau[23,0] )
 GC577: exp_tau_var_23_1 = EXP ( tau[23,1] )
 GC578: exp_tau_var_23_2 = EXP ( tau[23,2] )
 GC579: exp_tau_var_23_3 = EXP ( tau[23,3] )
 GC580: exp_tau_var_23_4 = EXP ( tau[23,4] )
 GC581: exp_tau_var_23_5 = EXP ( tau[23,5] )
 GC582: exp_tau_var_23_6 = EXP ( tau[23,6] )
 GC583: exp_tau_var_23_7 = EXP ( tau[23,7] )
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
 GC601: exp_tau_var_24_0 = EXP ( tau[24,0] )
 GC602: exp_tau_var_24_1 = EXP ( tau[24,1] )
 GC603: exp_tau_var_24_2 = EXP ( tau[24,2] )
 GC604: exp_tau_var_24_3 = EXP ( tau[24,3] )
 GC605: exp_tau_var_24_4 = EXP ( tau[24,4] )
 GC606: exp_tau_var_24_5 = EXP ( tau[24,5] )
 GC607: exp_tau_var_24_6 = EXP ( tau[24,6] )
 GC608: exp_tau_var_24_7 = EXP ( tau[24,7] )
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
