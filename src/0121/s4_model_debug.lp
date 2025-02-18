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
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper + 0 max_r_helper
   + 0 max_r_helper + 0 max_r_helper
Subject To
 R0: - 0.0398448308177865 alphas[0] - 0.3779089760840579 alphas[1]
   + 0.5400971290116704 alphas[2] - 1.778249581914427 alphas[3] - alphas[4]
   + f_var[0] = 0
 Q0_upper_bound_0: - 2028.416054007841 Fraction_for_second_order_amount[0]
   + Q0_var[0] = 0
 tau_computation_0_0: - betas[0] - 0.0398448308177865 gammas[0]
   - 0.3779089760840579 gammas[1] + 0.5400971290116704 gammas[2]
   - 1.778249581914427 gammas[3] + tau[0,0] = 0
 tau_computation_0_1: - betas[1] - 0.0398448308177865 gammas[0]
   - 0.3779089760840579 gammas[1] + 0.5400971290116704 gammas[2]
   - 1.778249581914427 gammas[3] + tau[0,1] = 0
 tau_computation_0_2: - betas[2] - 0.0398448308177865 gammas[0]
   - 0.3779089760840579 gammas[1] + 0.5400971290116704 gammas[2]
   - 1.778249581914427 gammas[3] + tau[0,2] = 0
 tau_computation_0_3: - betas[3] - 0.0398448308177865 gammas[0]
   - 0.3779089760840579 gammas[1] + 0.5400971290116704 gammas[2]
   - 1.778249581914427 gammas[3] + tau[0,3] = 0
 tau_computation_0_4: - betas[4] - 0.0398448308177865 gammas[0]
   - 0.3779089760840579 gammas[1] + 0.5400971290116704 gammas[2]
   - 1.778249581914427 gammas[3] + tau[0,4] = 0
 tau_computation_0_5: - betas[5] - 0.0398448308177865 gammas[0]
   - 0.3779089760840579 gammas[1] + 0.5400971290116704 gammas[2]
   - 1.778249581914427 gammas[3] + tau[0,5] = 0
 tau_computation_0_6: - betas[6] - 0.0398448308177865 gammas[0]
   - 0.3779089760840579 gammas[1] + 0.5400971290116704 gammas[2]
   - 1.778249581914427 gammas[3] + tau[0,6] = 0
 tau_computation_0_7: - betas[7] - 0.0398448308177865 gammas[0]
   - 0.3779089760840579 gammas[1] + 0.5400971290116704 gammas[2]
   - 1.778249581914427 gammas[3] + tau[0,7] = 0
 Ensure_only_one_R_true_0: R[0,0] + R[0,1] + R[0,2] + R[0,3] + R[0,4]
   + R[0,5] + R[0,6] + R[0,7] = 1
 Define_Q_hat_0: - 1541.976734981009 R[0,0] - 1340.921900475007 R[0,1]
   - 1281.756481531044 R[0,2] - 1285.32955404119 R[0,3]
   - 1498.220605072307 R[0,4] - 1498.220541768304 R[0,5]
   - 1316.293666105063 R[0,6] - 1241.399889064605 R[0,7] + Q_hat[0] = 0
 Adjust_Q_hat_0: Q0_var[0] - Q_hat[0] + Q_hat_adjusted[0] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_0: Total_Demand_Up_to_K_minus_1[0]
   - 42.53400712928162 R[0,0] - 170.4774610776255 R[0,1]
   - 170.4774610776255 R[0,2] - 298.426431203839 R[0,3]
   - 665.2888987848355 R[0,4] - 793.2378692833612 R[0,5]
   - 912.2337328643589 R[0,6] - 985.5016692210908 R[0,7] = 0
 Constr_Total_Demand_from_K_to_T_0: Total_Demand_from_k_to_T[0]
   - 1198.865603088862 R[0,0] - 1070.922149140518 R[0,1]
   - 1070.922149140519 R[0,2] - 942.9731790143049 R[0,3]
   - 576.1107114333083 R[0,4] - 448.1617409347827 R[0,5]
   - 329.165877353785 R[0,6] - 255.8979409970531 R[0,7] = 0
 Constr_Left_0_diff_aux_0: Sold_0[0] - Q0_var[0] + Left_0_aux_0 = 0
 Constr_Lost_0_diff_aux_0: - Total_Demand_Up_to_K_minus_1[0] + Q0_var[0]
   + Lost_0_aux_0 = 0
 Constr_Q1_plus_left_0: - Left_0[0] + Q1_plus_left[0] - Q1_var[0] = 0
 Constr_Left_1_diff_aux_0: Sold_1[0] - Q1_plus_left[0] + Left_0_aux_0 = 0
 Constr_Lost_1_diff_aux_0: - Total_Demand_from_k_to_T[0] + Q1_plus_left[0]
   + Lost_0_aux_0 = 0
 Profit_Constraint_0: - 700 Sold_0[0] + 700 Lost_0[0] - 700 Sold_1[0]
   + 300 Left_1[0] + 700 Lost_1[0] + profits_vars[0] = 0
 R21: - 0.6509235944307136 alphas[0] + 1.28709053145843 alphas[1]
   - 0.856977027260221 alphas[2] - 1.297297962226022 alphas[3] - alphas[4]
   + f_var[1] = 0
 Q0_upper_bound_1: - 2028.416054007841 Fraction_for_second_order_amount[1]
   + Q0_var[1] = 0
 tau_computation_1_0: - betas[0] - 0.6509235944307136 gammas[0]
   + 1.28709053145843 gammas[1] - 0.856977027260221 gammas[2]
   - 1.297297962226022 gammas[3] + tau[1,0] = 0
 tau_computation_1_1: - betas[1] - 0.6509235944307136 gammas[0]
   + 1.28709053145843 gammas[1] - 0.856977027260221 gammas[2]
   - 1.297297962226022 gammas[3] + tau[1,1] = 0
 tau_computation_1_2: - betas[2] - 0.6509235944307136 gammas[0]
   + 1.28709053145843 gammas[1] - 0.856977027260221 gammas[2]
   - 1.297297962226022 gammas[3] + tau[1,2] = 0
 tau_computation_1_3: - betas[3] - 0.6509235944307136 gammas[0]
   + 1.28709053145843 gammas[1] - 0.856977027260221 gammas[2]
   - 1.297297962226022 gammas[3] + tau[1,3] = 0
 tau_computation_1_4: - betas[4] - 0.6509235944307136 gammas[0]
   + 1.28709053145843 gammas[1] - 0.856977027260221 gammas[2]
   - 1.297297962226022 gammas[3] + tau[1,4] = 0
 tau_computation_1_5: - betas[5] - 0.6509235944307136 gammas[0]
   + 1.28709053145843 gammas[1] - 0.856977027260221 gammas[2]
   - 1.297297962226022 gammas[3] + tau[1,5] = 0
 tau_computation_1_6: - betas[6] - 0.6509235944307136 gammas[0]
   + 1.28709053145843 gammas[1] - 0.856977027260221 gammas[2]
   - 1.297297962226022 gammas[3] + tau[1,6] = 0
 tau_computation_1_7: - betas[7] - 0.6509235944307136 gammas[0]
   + 1.28709053145843 gammas[1] - 0.856977027260221 gammas[2]
   - 1.297297962226022 gammas[3] + tau[1,7] = 0
 Ensure_only_one_R_true_1: R[1,0] + R[1,1] + R[1,2] + R[1,3] + R[1,4]
   + R[1,5] + R[1,6] + R[1,7] = 1
 Define_Q_hat_1: - 2109.413863932218 R[1,0] - 2318.485692626555 R[1,1]
   - 2259.87339224262 R[1,2] - 2270.351244466319 R[1,3]
   - 2191.524363408369 R[1,4] - 2191.524402660232 R[1,5]
   - 2136.41042568206 R[1,6] - 2018.892180807597 R[1,7] + Q_hat[1] = 0
 Adjust_Q_hat_1: Q0_var[1] - Q_hat[1] + Q_hat_adjusted[1] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_1: Total_Demand_Up_to_K_minus_1[1]
   - 225.614664394751 R[1,0] - 449.7005355513063 R[1,1]
   - 579.8271664686233 R[1,2] - 803.91303845681 R[1,3]
   - 1002.631684989354 R[1,4] - 1226.717557193219 R[1,5]
   - 1485.590959638209 R[1,6] - 1570.720159318349 R[1,7] = 0
 Constr_Total_Demand_from_K_to_T_1: Total_Demand_from_k_to_T[1]
   - 1793.277239331328 R[1,0] - 1569.191368174772 R[1,1]
   - 1439.064737257455 R[1,2] - 1214.978865269268 R[1,3]
   - 1016.260218736724 R[1,4] - 792.1743465328593 R[1,5]
   - 533.3009440878691 R[1,6] - 448.1717444077298 R[1,7] = 0
 Constr_Left_0_diff_aux_1: Sold_0[1] - Q0_var[1] + Left_0_aux_1 = 0
 Constr_Lost_0_diff_aux_1: - Total_Demand_Up_to_K_minus_1[1] + Q0_var[1]
   + Lost_0_aux_1 = 0
 Constr_Q1_plus_left_1: - Left_0[1] + Q1_plus_left[1] - Q1_var[1] = 0
 Constr_Left_1_diff_aux_1: Sold_1[1] - Q1_plus_left[1] + Left_0_aux_1 = 0
 Constr_Lost_1_diff_aux_1: - Total_Demand_from_k_to_T[1] + Q1_plus_left[1]
   + Lost_0_aux_1 = 0
 Profit_Constraint_1: - 700 Sold_0[1] + 700 Lost_0[1] - 700 Sold_1[1]
   + 300 Left_1[1] + 700 Lost_1[1] + profits_vars[1] = 0
 R42: - 0.2379962975181053 alphas[0] - 0.6121695908395033 alphas[1]
   + 0.3004769577787669 alphas[2] + 0.3030005211230327 alphas[3]
   - alphas[4] + f_var[2] = 0
 Q0_upper_bound_2: - 2028.416054007841 Fraction_for_second_order_amount[2]
   + Q0_var[2] = 0
 tau_computation_2_0: - betas[0] - 0.2379962975181053 gammas[0]
   - 0.6121695908395033 gammas[1] + 0.3004769577787669 gammas[2]
   + 0.3030005211230327 gammas[3] + tau[2,0] = 0
 tau_computation_2_1: - betas[1] - 0.2379962975181053 gammas[0]
   - 0.6121695908395033 gammas[1] + 0.3004769577787669 gammas[2]
   + 0.3030005211230327 gammas[3] + tau[2,1] = 0
 tau_computation_2_2: - betas[2] - 0.2379962975181053 gammas[0]
   - 0.6121695908395033 gammas[1] + 0.3004769577787669 gammas[2]
   + 0.3030005211230327 gammas[3] + tau[2,2] = 0
 tau_computation_2_3: - betas[3] - 0.2379962975181053 gammas[0]
   - 0.6121695908395033 gammas[1] + 0.3004769577787669 gammas[2]
   + 0.3030005211230327 gammas[3] + tau[2,3] = 0
 tau_computation_2_4: - betas[4] - 0.2379962975181053 gammas[0]
   - 0.6121695908395033 gammas[1] + 0.3004769577787669 gammas[2]
   + 0.3030005211230327 gammas[3] + tau[2,4] = 0
 tau_computation_2_5: - betas[5] - 0.2379962975181053 gammas[0]
   - 0.6121695908395033 gammas[1] + 0.3004769577787669 gammas[2]
   + 0.3030005211230327 gammas[3] + tau[2,5] = 0
 tau_computation_2_6: - betas[6] - 0.2379962975181053 gammas[0]
   - 0.6121695908395033 gammas[1] + 0.3004769577787669 gammas[2]
   + 0.3030005211230327 gammas[3] + tau[2,6] = 0
 tau_computation_2_7: - betas[7] - 0.2379962975181053 gammas[0]
   - 0.6121695908395033 gammas[1] + 0.3004769577787669 gammas[2]
   + 0.3030005211230327 gammas[3] + tau[2,7] = 0
 Ensure_only_one_R_true_2: R[2,0] + R[2,1] + R[2,2] + R[2,3] + R[2,4]
   + R[2,5] + R[2,6] + R[2,7] = 1
 Define_Q_hat_2: - 2467.078074384144 R[2,0] - 1656.66757569297 R[2,1]
   - 1650.456479500465 R[2,2] - 1645.080831656612 R[2,3]
   - 1551.15759365846 R[2,4] - 1551.157568290426 R[2,5]
   - 1369.359069826195 R[2,6] - 1312.896117960501 R[2,7] + Q_hat[2] = 0
 Adjust_Q_hat_2: Q0_var[2] - Q_hat[2] + Q_hat_adjusted[2] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_2: Total_Demand_Up_to_K_minus_1[2]
   - 341.0131735668643 R[2,0] - 485.455950372997 R[2,1]
   - 717.209647072978 R[2,2] - 861.6476069080037 R[2,3]
   - 861.6476069080037 R[2,4] - 1006.085566731124 R[2,5]
   - 1006.085566731124 R[2,6] - 1024.019920128306 R[2,7] = 0
 Constr_Total_Demand_from_K_to_T_2: Total_Demand_from_k_to_T[2]
   - 971.8826662076817 R[2,0] - 827.4398894015492 R[2,1]
   - 595.6861927015681 R[2,2] - 451.2482328665424 R[2,3]
   - 451.2482328665424 R[2,4] - 306.8102730434222 R[2,5]
   - 306.8102730434222 R[2,6] - 288.8759196462405 R[2,7] = 0
 Constr_Left_0_diff_aux_2: Sold_0[2] - Q0_var[2] + Left_0_aux_2 = 0
 Constr_Lost_0_diff_aux_2: - Total_Demand_Up_to_K_minus_1[2] + Q0_var[2]
   + Lost_0_aux_2 = 0
 Constr_Q1_plus_left_2: - Left_0[2] + Q1_plus_left[2] - Q1_var[2] = 0
 Constr_Left_1_diff_aux_2: Sold_1[2] - Q1_plus_left[2] + Left_0_aux_2 = 0
 Constr_Lost_1_diff_aux_2: - Total_Demand_from_k_to_T[2] + Q1_plus_left[2]
   + Lost_0_aux_2 = 0
 Profit_Constraint_2: - 700 Sold_0[2] + 700 Lost_0[2] - 700 Sold_1[2]
   + 300 Left_1[2] + 700 Lost_1[2] + profits_vars[2] = 0
 R63: - 0.0254092292859822 alphas[0] - 1.538942107976239 alphas[1]
   + 1.020355773715186 alphas[2] + 1.217818518889134 alphas[3] - alphas[4]
   + f_var[3] = 0
 Q0_upper_bound_3: - 2028.416054007841 Fraction_for_second_order_amount[3]
   + Q0_var[3] = 0
 tau_computation_3_0: - betas[0] - 0.0254092292859822 gammas[0]
   - 1.538942107976239 gammas[1] + 1.020355773715186 gammas[2]
   + 1.217818518889134 gammas[3] + tau[3,0] = 0
 tau_computation_3_1: - betas[1] - 0.0254092292859822 gammas[0]
   - 1.538942107976239 gammas[1] + 1.020355773715186 gammas[2]
   + 1.217818518889134 gammas[3] + tau[3,1] = 0
 tau_computation_3_2: - betas[2] - 0.0254092292859822 gammas[0]
   - 1.538942107976239 gammas[1] + 1.020355773715186 gammas[2]
   + 1.217818518889134 gammas[3] + tau[3,2] = 0
 tau_computation_3_3: - betas[3] - 0.0254092292859822 gammas[0]
   - 1.538942107976239 gammas[1] + 1.020355773715186 gammas[2]
   + 1.217818518889134 gammas[3] + tau[3,3] = 0
 tau_computation_3_4: - betas[4] - 0.0254092292859822 gammas[0]
   - 1.538942107976239 gammas[1] + 1.020355773715186 gammas[2]
   + 1.217818518889134 gammas[3] + tau[3,4] = 0
 tau_computation_3_5: - betas[5] - 0.0254092292859822 gammas[0]
   - 1.538942107976239 gammas[1] + 1.020355773715186 gammas[2]
   + 1.217818518889134 gammas[3] + tau[3,5] = 0
 tau_computation_3_6: - betas[6] - 0.0254092292859822 gammas[0]
   - 1.538942107976239 gammas[1] + 1.020355773715186 gammas[2]
   + 1.217818518889134 gammas[3] + tau[3,6] = 0
 tau_computation_3_7: - betas[7] - 0.0254092292859822 gammas[0]
   - 1.538942107976239 gammas[1] + 1.020355773715186 gammas[2]
   + 1.217818518889134 gammas[3] + tau[3,7] = 0
 Ensure_only_one_R_true_3: R[3,0] + R[3,1] + R[3,2] + R[3,3] + R[3,4]
   + R[3,5] + R[3,6] + R[3,7] = 1
 Define_Q_hat_3: - 1720.714106246457 R[3,0] - 1072.423894217563 R[3,1]
   - 1050.211679583559 R[3,2] - 1063.756046670566 R[3,3]
   - 905.1746892002971 R[3,4] - 905.1746212867117 R[3,5]
   - 1024.555512681005 R[3,6] - 1031.663543461739 R[3,7] + Q_hat[3] = 0
 Adjust_Q_hat_3: Q0_var[3] - Q_hat[3] + Q_hat_adjusted[3] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_3: Total_Demand_Up_to_K_minus_1[3]
   - 100.2026962925424 R[3,0] - 195.1752718971154 R[3,1]
   - 271.6651355850347 R[3,2] - 366.5660545736286 R[3,3]
   - 366.5660545736286 R[3,4] - 461.4669739635416 R[3,5]
   - 682.1688975310507 R[3,6] - 841.8614268697045 R[3,7] = 0
 Constr_Total_Demand_from_K_to_T_3: Total_Demand_from_k_to_T[3]
   - 931.4605693555766 R[3,0] - 836.4879937510036 R[3,1]
   - 759.9981300630843 R[3,2] - 665.0972110744904 R[3,3]
   - 665.0972110744904 R[3,4] - 570.1962916845773 R[3,5]
   - 349.4943681170682 R[3,6] - 189.8018387784145 R[3,7] = 0
 Constr_Left_0_diff_aux_3: Sold_0[3] - Q0_var[3] + Left_0_aux_3 = 0
 Constr_Lost_0_diff_aux_3: - Total_Demand_Up_to_K_minus_1[3] + Q0_var[3]
   + Lost_0_aux_3 = 0
 Constr_Q1_plus_left_3: - Left_0[3] + Q1_plus_left[3] - Q1_var[3] = 0
 Constr_Left_1_diff_aux_3: Sold_1[3] - Q1_plus_left[3] + Left_0_aux_3 = 0
 Constr_Lost_1_diff_aux_3: - Total_Demand_from_k_to_T[3] + Q1_plus_left[3]
   + Lost_0_aux_3 = 0
 Profit_Constraint_3: - 700 Sold_0[3] + 700 Lost_0[3] - 700 Sold_1[3]
   + 300 Left_1[3] + 700 Lost_1[3] + profits_vars[3] = 0
 R84: 0.4198482119787716 alphas[0] + 1.373869682500199 alphas[1]
   + 1.583203230845045 alphas[2] + 0.0340616211972166 alphas[3] - alphas[4]
   + f_var[4] = 0
 Q0_upper_bound_4: - 2028.416054007841 Fraction_for_second_order_amount[4]
   + Q0_var[4] = 0
 tau_computation_4_0: - betas[0] + 0.4198482119787716 gammas[0]
   + 1.373869682500199 gammas[1] + 1.583203230845045 gammas[2]
   + 0.0340616211972166 gammas[3] + tau[4,0] = 0
 tau_computation_4_1: - betas[1] + 0.4198482119787716 gammas[0]
   + 1.373869682500199 gammas[1] + 1.583203230845045 gammas[2]
   + 0.0340616211972166 gammas[3] + tau[4,1] = 0
 tau_computation_4_2: - betas[2] + 0.4198482119787716 gammas[0]
   + 1.373869682500199 gammas[1] + 1.583203230845045 gammas[2]
   + 0.0340616211972166 gammas[3] + tau[4,2] = 0
 tau_computation_4_3: - betas[3] + 0.4198482119787716 gammas[0]
   + 1.373869682500199 gammas[1] + 1.583203230845045 gammas[2]
   + 0.0340616211972166 gammas[3] + tau[4,3] = 0
 tau_computation_4_4: - betas[4] + 0.4198482119787716 gammas[0]
   + 1.373869682500199 gammas[1] + 1.583203230845045 gammas[2]
   + 0.0340616211972166 gammas[3] + tau[4,4] = 0
 tau_computation_4_5: - betas[5] + 0.4198482119787716 gammas[0]
   + 1.373869682500199 gammas[1] + 1.583203230845045 gammas[2]
   + 0.0340616211972166 gammas[3] + tau[4,5] = 0
 tau_computation_4_6: - betas[6] + 0.4198482119787716 gammas[0]
   + 1.373869682500199 gammas[1] + 1.583203230845045 gammas[2]
   + 0.0340616211972166 gammas[3] + tau[4,6] = 0
 tau_computation_4_7: - betas[7] + 0.4198482119787716 gammas[0]
   + 1.373869682500199 gammas[1] + 1.583203230845045 gammas[2]
   + 0.0340616211972166 gammas[3] + tau[4,7] = 0
 Ensure_only_one_R_true_4: R[4,0] + R[4,1] + R[4,2] + R[4,3] + R[4,4]
   + R[4,5] + R[4,6] + R[4,7] = 1
 Define_Q_hat_4: - 2056.347216144249 R[4,0] - 782.8157876088718 R[4,1]
   - 847.5714478677671 R[4,2] - 935.741030209856 R[4,3]
   - 1146.870250731055 R[4,4] - 1146.870713876878 R[4,5]
   - 1071.835586483615 R[4,6] - 1056.997704564391 R[4,7] + Q_hat[4] = 0
 Adjust_Q_hat_4: Q0_var[4] - Q_hat[4] + Q_hat_adjusted[4] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_4: Total_Demand_Up_to_K_minus_1[4]
   - 208.4929847227606 R[4,0] - 265.0999586506043 R[4,1]
   - 534.6738413787868 R[4,2] - 590.8435232289357 R[4,3]
   - 750.883471014595 R[4,4] - 807.0531408826463 R[4,5]
   - 807.0531408826463 R[4,6] - 944.6580682618797 R[4,7] = 0
 Constr_Total_Demand_from_K_to_T_4: Total_Demand_from_k_to_T[4]
   - 848.5044477346567 R[4,0] - 791.897473806813 R[4,1]
   - 522.3235910786304 R[4,2] - 466.1539092284816 R[4,3]
   - 306.1139614428224 R[4,4] - 249.9442915747711 R[4,5]
   - 249.9442915747711 R[4,6] - 112.3393641955375 R[4,7] = 0
 Constr_Left_0_diff_aux_4: Sold_0[4] - Q0_var[4] + Left_0_aux_4 = 0
 Constr_Lost_0_diff_aux_4: - Total_Demand_Up_to_K_minus_1[4] + Q0_var[4]
   + Lost_0_aux_4 = 0
 Constr_Q1_plus_left_4: - Left_0[4] + Q1_plus_left[4] - Q1_var[4] = 0
 Constr_Left_1_diff_aux_4: Sold_1[4] - Q1_plus_left[4] + Left_0_aux_4 = 0
 Constr_Lost_1_diff_aux_4: - Total_Demand_from_k_to_T[4] + Q1_plus_left[4]
   + Lost_0_aux_4 = 0
 Profit_Constraint_4: - 700 Sold_0[4] + 700 Lost_0[4] - 700 Sold_1[4]
   + 300 Left_1[4] + 700 Lost_1[4] + profits_vars[4] = 0
 R105: - 0.3964103670188668 alphas[0] + 0.0371448923323029 alphas[1]
   + 1.428541733969417 alphas[2] + 0.6692671061127012 alphas[3] - alphas[4]
   + f_var[5] = 0
 Q0_upper_bound_5: - 2028.416054007841 Fraction_for_second_order_amount[5]
   + Q0_var[5] = 0
 tau_computation_5_0: - betas[0] - 0.3964103670188668 gammas[0]
   + 0.0371448923323029 gammas[1] + 1.428541733969417 gammas[2]
   + 0.6692671061127012 gammas[3] + tau[5,0] = 0
 tau_computation_5_1: - betas[1] - 0.3964103670188668 gammas[0]
   + 0.0371448923323029 gammas[1] + 1.428541733969417 gammas[2]
   + 0.6692671061127012 gammas[3] + tau[5,1] = 0
 tau_computation_5_2: - betas[2] - 0.3964103670188668 gammas[0]
   + 0.0371448923323029 gammas[1] + 1.428541733969417 gammas[2]
   + 0.6692671061127012 gammas[3] + tau[5,2] = 0
 tau_computation_5_3: - betas[3] - 0.3964103670188668 gammas[0]
   + 0.0371448923323029 gammas[1] + 1.428541733969417 gammas[2]
   + 0.6692671061127012 gammas[3] + tau[5,3] = 0
 tau_computation_5_4: - betas[4] - 0.3964103670188668 gammas[0]
   + 0.0371448923323029 gammas[1] + 1.428541733969417 gammas[2]
   + 0.6692671061127012 gammas[3] + tau[5,4] = 0
 tau_computation_5_5: - betas[5] - 0.3964103670188668 gammas[0]
   + 0.0371448923323029 gammas[1] + 1.428541733969417 gammas[2]
   + 0.6692671061127012 gammas[3] + tau[5,5] = 0
 tau_computation_5_6: - betas[6] - 0.3964103670188668 gammas[0]
   + 0.0371448923323029 gammas[1] + 1.428541733969417 gammas[2]
   + 0.6692671061127012 gammas[3] + tau[5,6] = 0
 tau_computation_5_7: - betas[7] - 0.3964103670188668 gammas[0]
   + 0.0371448923323029 gammas[1] + 1.428541733969417 gammas[2]
   + 0.6692671061127012 gammas[3] + tau[5,7] = 0
 Ensure_only_one_R_true_5: R[5,0] + R[5,1] + R[5,2] + R[5,3] + R[5,4]
   + R[5,5] + R[5,6] + R[5,7] = 1
 Define_Q_hat_5: - 1872.512475166899 R[5,0] - 845.923295078212 R[5,1]
   - 847.4705121740819 R[5,2] - 920.4883428260408 R[5,3]
   - 832.005238139822 R[5,4] - 832.0049464138907 R[5,5]
   - 806.1513474060012 R[5,6] - 755.1638181964206 R[5,7] + Q_hat[5] = 0
 Adjust_Q_hat_5: Q0_var[5] - Q_hat[5] + Q_hat_adjusted[5] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_5: Total_Demand_Up_to_K_minus_1[5]
   - 149.1796555640433 R[5,0] - 216.3185484551709 R[5,1]
   - 342.5910387468933 R[5,2] - 409.403446630448 R[5,3]
   - 409.403446630448 R[5,4] - 476.2158544962552 R[5,5]
   - 561.4902910159285 R[5,6] - 621.5387210066472 R[5,7] = 0
 Constr_Total_Demand_from_K_to_T_5: Total_Demand_from_k_to_T[5]
   - 605.9838811740871 R[5,0] - 538.8449882829595 R[5,1]
   - 412.5724979912371 R[5,2] - 345.7600901076825 R[5,3]
   - 345.7600901076825 R[5,4] - 278.9476822418753 R[5,5]
   - 193.6732457222019 R[5,6] - 133.6248157314834 R[5,7] = 0
 Constr_Left_0_diff_aux_5: Sold_0[5] - Q0_var[5] + Left_0_aux_5 = 0
 Constr_Lost_0_diff_aux_5: - Total_Demand_Up_to_K_minus_1[5] + Q0_var[5]
   + Lost_0_aux_5 = 0
 Constr_Q1_plus_left_5: - Left_0[5] + Q1_plus_left[5] - Q1_var[5] = 0
 Constr_Left_1_diff_aux_5: Sold_1[5] - Q1_plus_left[5] + Left_0_aux_5 = 0
 Constr_Lost_1_diff_aux_5: - Total_Demand_from_k_to_T[5] + Q1_plus_left[5]
   + Lost_0_aux_5 = 0
 Profit_Constraint_5: - 700 Sold_0[5] + 700 Lost_0[5] - 700 Sold_1[5]
   + 300 Left_1[5] + 700 Lost_1[5] + profits_vars[5] = 0
 R126: 0.3686761193000995 alphas[0] - 0.027708689224906 alphas[1]
   - 0.79539451867227 alphas[2] + 1.03130701321808 alphas[3] - alphas[4]
   + f_var[6] = 0
 Q0_upper_bound_6: - 2028.416054007841 Fraction_for_second_order_amount[6]
   + Q0_var[6] = 0
 tau_computation_6_0: - betas[0] + 0.3686761193000995 gammas[0]
   - 0.027708689224906 gammas[1] - 0.79539451867227 gammas[2]
   + 1.03130701321808 gammas[3] + tau[6,0] = 0
 tau_computation_6_1: - betas[1] + 0.3686761193000995 gammas[0]
   - 0.027708689224906 gammas[1] - 0.79539451867227 gammas[2]
   + 1.03130701321808 gammas[3] + tau[6,1] = 0
 tau_computation_6_2: - betas[2] + 0.3686761193000995 gammas[0]
   - 0.027708689224906 gammas[1] - 0.79539451867227 gammas[2]
   + 1.03130701321808 gammas[3] + tau[6,2] = 0
 tau_computation_6_3: - betas[3] + 0.3686761193000995 gammas[0]
   - 0.027708689224906 gammas[1] - 0.79539451867227 gammas[2]
   + 1.03130701321808 gammas[3] + tau[6,3] = 0
 tau_computation_6_4: - betas[4] + 0.3686761193000995 gammas[0]
   - 0.027708689224906 gammas[1] - 0.79539451867227 gammas[2]
   + 1.03130701321808 gammas[3] + tau[6,4] = 0
 tau_computation_6_5: - betas[5] + 0.3686761193000995 gammas[0]
   - 0.027708689224906 gammas[1] - 0.79539451867227 gammas[2]
   + 1.03130701321808 gammas[3] + tau[6,5] = 0
 tau_computation_6_6: - betas[6] + 0.3686761193000995 gammas[0]
   - 0.027708689224906 gammas[1] - 0.79539451867227 gammas[2]
   + 1.03130701321808 gammas[3] + tau[6,6] = 0
 tau_computation_6_7: - betas[7] + 0.3686761193000995 gammas[0]
   - 0.027708689224906 gammas[1] - 0.79539451867227 gammas[2]
   + 1.03130701321808 gammas[3] + tau[6,7] = 0
 Ensure_only_one_R_true_6: R[6,0] + R[6,1] + R[6,2] + R[6,3] + R[6,4]
   + R[6,5] + R[6,6] + R[6,7] = 1
 Define_Q_hat_6: - 2268.690833809383 R[6,0] - 2308.345160126873 R[6,1]
   - 2316.279963173551 R[6,2] - 2312.802783828587 R[6,3]
   - 2169.013854945765 R[6,4] - 2169.01392904271 R[6,5]
   - 2225.456716871219 R[6,6] - 2286.92635238352 R[6,7] + Q_hat[6] = 0
 Adjust_Q_hat_6: Q0_var[6] - Q_hat[6] + Q_hat_adjusted[6] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_6: Total_Demand_Up_to_K_minus_1[6]
   - 277.0045556171689 R[6,0] - 496.8527546810832 R[6,1]
   - 777.7038403397735 R[6,2] - 997.5520342169043 R[6,3]
   - 1081.217781384751 R[6,4] - 1301.065974759392 R[6,5]
   - 1537.146452804285 R[6,6] - 1847.22968764704 R[6,7] = 0
 Constr_Total_Demand_from_K_to_T_6: Total_Demand_from_k_to_T[6]
   - 2009.921518779154 R[6,0] - 1790.07331971524 R[6,1]
   - 1509.22223405655 R[6,2] - 1289.374040179419 R[6,3]
   - 1205.708293011573 R[6,4] - 985.8600996369308 R[6,5]
   - 749.7796215920384 R[6,6] - 439.6963867492836 R[6,7] = 0
 Constr_Left_0_diff_aux_6: Sold_0[6] - Q0_var[6] + Left_0_aux_6 = 0
 Constr_Lost_0_diff_aux_6: - Total_Demand_Up_to_K_minus_1[6] + Q0_var[6]
   + Lost_0_aux_6 = 0
 Constr_Q1_plus_left_6: - Left_0[6] + Q1_plus_left[6] - Q1_var[6] = 0
 Constr_Left_1_diff_aux_6: Sold_1[6] - Q1_plus_left[6] + Left_0_aux_6 = 0
 Constr_Lost_1_diff_aux_6: - Total_Demand_from_k_to_T[6] + Q1_plus_left[6]
   + Lost_0_aux_6 = 0
 Profit_Constraint_6: - 700 Sold_0[6] + 700 Lost_0[6] - 700 Sold_1[6]
   + 300 Left_1[6] + 700 Lost_1[6] + profits_vars[6] = 0
 R147: - 1.299494290945913 alphas[0] + 0.0530899639197399 alphas[1]
   + 0.0245662453739224 alphas[2] + 1.318264820481297 alphas[3] - alphas[4]
   + f_var[7] = 0
 Q0_upper_bound_7: - 2028.416054007841 Fraction_for_second_order_amount[7]
   + Q0_var[7] = 0
 tau_computation_7_0: - betas[0] - 1.299494290945913 gammas[0]
   + 0.0530899639197399 gammas[1] + 0.0245662453739224 gammas[2]
   + 1.318264820481297 gammas[3] + tau[7,0] = 0
 tau_computation_7_1: - betas[1] - 1.299494290945913 gammas[0]
   + 0.0530899639197399 gammas[1] + 0.0245662453739224 gammas[2]
   + 1.318264820481297 gammas[3] + tau[7,1] = 0
 tau_computation_7_2: - betas[2] - 1.299494290945913 gammas[0]
   + 0.0530899639197399 gammas[1] + 0.0245662453739224 gammas[2]
   + 1.318264820481297 gammas[3] + tau[7,2] = 0
 tau_computation_7_3: - betas[3] - 1.299494290945913 gammas[0]
   + 0.0530899639197399 gammas[1] + 0.0245662453739224 gammas[2]
   + 1.318264820481297 gammas[3] + tau[7,3] = 0
 tau_computation_7_4: - betas[4] - 1.299494290945913 gammas[0]
   + 0.0530899639197399 gammas[1] + 0.0245662453739224 gammas[2]
   + 1.318264820481297 gammas[3] + tau[7,4] = 0
 tau_computation_7_5: - betas[5] - 1.299494290945913 gammas[0]
   + 0.0530899639197399 gammas[1] + 0.0245662453739224 gammas[2]
   + 1.318264820481297 gammas[3] + tau[7,5] = 0
 tau_computation_7_6: - betas[6] - 1.299494290945913 gammas[0]
   + 0.0530899639197399 gammas[1] + 0.0245662453739224 gammas[2]
   + 1.318264820481297 gammas[3] + tau[7,6] = 0
 tau_computation_7_7: - betas[7] - 1.299494290945913 gammas[0]
   + 0.0530899639197399 gammas[1] + 0.0245662453739224 gammas[2]
   + 1.318264820481297 gammas[3] + tau[7,7] = 0
 Ensure_only_one_R_true_7: R[7,0] + R[7,1] + R[7,2] + R[7,3] + R[7,4]
   + R[7,5] + R[7,6] + R[7,7] = 1
 Define_Q_hat_7: - 1531.066923105586 R[7,0] - 1662.2969765947 R[7,1]
   - 1616.415785847372 R[7,2] - 1621.407625330897 R[7,3]
   - 1448.499036252263 R[7,4] - 1448.499097444089 R[7,5]
   - 1386.385511380958 R[7,6] - 1369.230483017808 R[7,7] + Q_hat[7] = 0
 Adjust_Q_hat_7: Q0_var[7] - Q_hat[7] + Q_hat_adjusted[7] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_7: Total_Demand_Up_to_K_minus_1[7]
   - 39.01401270719147 R[7,0] - 202.4382889004705 R[7,1]
   - 249.1306549592395 R[7,2] - 412.554865849218 R[7,3]
   - 529.1682020417497 R[7,4] - 692.592413180761 R[7,5]
   - 868.8594689760347 R[7,6] - 1042.381783858741 R[7,7] = 0
 Constr_Total_Demand_from_K_to_T_7: Total_Demand_from_k_to_T[7]
   - 1330.216193429572 R[7,0] - 1166.791917236293 R[7,1]
   - 1120.099551177524 R[7,2] - 956.6753402875459 R[7,3]
   - 840.0620040950139 R[7,4] - 676.6377929560026 R[7,5]
   - 500.370737160729 R[7,6] - 326.8484222780227 R[7,7] = 0
 Constr_Left_0_diff_aux_7: Sold_0[7] - Q0_var[7] + Left_0_aux_7 = 0
 Constr_Lost_0_diff_aux_7: - Total_Demand_Up_to_K_minus_1[7] + Q0_var[7]
   + Lost_0_aux_7 = 0
 Constr_Q1_plus_left_7: - Left_0[7] + Q1_plus_left[7] - Q1_var[7] = 0
 Constr_Left_1_diff_aux_7: Sold_1[7] - Q1_plus_left[7] + Left_0_aux_7 = 0
 Constr_Lost_1_diff_aux_7: - Total_Demand_from_k_to_T[7] + Q1_plus_left[7]
   + Lost_0_aux_7 = 0
 Profit_Constraint_7: - 700 Sold_0[7] + 700 Lost_0[7] - 700 Sold_1[7]
   + 300 Left_1[7] + 700 Lost_1[7] + profits_vars[7] = 0
 R168: - 1.563536831972461 alphas[0] + 1.478681245539053 alphas[1]
   - 0.2765481297187581 alphas[2] - 1.121433075671428 alphas[3] - alphas[4]
   + f_var[8] = 0
 Q0_upper_bound_8: - 2028.416054007841 Fraction_for_second_order_amount[8]
   + Q0_var[8] = 0
 tau_computation_8_0: - betas[0] - 1.563536831972461 gammas[0]
   + 1.478681245539053 gammas[1] - 0.2765481297187581 gammas[2]
   - 1.121433075671428 gammas[3] + tau[8,0] = 0
 tau_computation_8_1: - betas[1] - 1.563536831972461 gammas[0]
   + 1.478681245539053 gammas[1] - 0.2765481297187581 gammas[2]
   - 1.121433075671428 gammas[3] + tau[8,1] = 0
 tau_computation_8_2: - betas[2] - 1.563536831972461 gammas[0]
   + 1.478681245539053 gammas[1] - 0.2765481297187581 gammas[2]
   - 1.121433075671428 gammas[3] + tau[8,2] = 0
 tau_computation_8_3: - betas[3] - 1.563536831972461 gammas[0]
   + 1.478681245539053 gammas[1] - 0.2765481297187581 gammas[2]
   - 1.121433075671428 gammas[3] + tau[8,3] = 0
 tau_computation_8_4: - betas[4] - 1.563536831972461 gammas[0]
   + 1.478681245539053 gammas[1] - 0.2765481297187581 gammas[2]
   - 1.121433075671428 gammas[3] + tau[8,4] = 0
 tau_computation_8_5: - betas[5] - 1.563536831972461 gammas[0]
   + 1.478681245539053 gammas[1] - 0.2765481297187581 gammas[2]
   - 1.121433075671428 gammas[3] + tau[8,5] = 0
 tau_computation_8_6: - betas[6] - 1.563536831972461 gammas[0]
   + 1.478681245539053 gammas[1] - 0.2765481297187581 gammas[2]
   - 1.121433075671428 gammas[3] + tau[8,6] = 0
 tau_computation_8_7: - betas[7] - 1.563536831972461 gammas[0]
   + 1.478681245539053 gammas[1] - 0.2765481297187581 gammas[2]
   - 1.121433075671428 gammas[3] + tau[8,7] = 0
 Ensure_only_one_R_true_8: R[8,0] + R[8,1] + R[8,2] + R[8,3] + R[8,4]
   + R[8,5] + R[8,6] + R[8,7] = 1
 Define_Q_hat_8: - 2267.794194419288 R[8,0] - 1982.830873335498 R[8,1]
   - 1947.375645403198 R[8,2] - 1949.940926735034 R[8,3]
   - 1727.167079845633 R[8,4] - 1727.167103324782 R[8,5]
   - 1641.901573842957 R[8,6] - 1626.193413040286 R[8,7] + Q_hat[8] = 0
 Adjust_Q_hat_8: Q0_var[8] - Q_hat[8] + Q_hat_adjusted[8] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_8: Total_Demand_Up_to_K_minus_1[8]
   - 276.7152595515904 R[8,0] - 460.8600438009335 R[8,1]
   - 633.0713505665585 R[8,2] - 817.2161529241821 R[8,3]
   - 817.2161529241821 R[8,4] - 1001.360955701363 R[8,5]
   - 1141.867746461829 R[8,6] - 1257.903530849573 R[8,7] = 0
 Constr_Total_Demand_from_K_to_T_8: Total_Demand_from_k_to_T[8]
   - 1349.477876852344 R[8,0] - 1165.333092603001 R[8,1]
   - 993.1217858373757 R[8,2] - 808.9769834797521 R[8,3]
   - 808.9769834797521 R[8,4] - 624.8321807025716 R[8,5]
   - 484.3253899421048 R[8,6] - 368.2896055543611 R[8,7] = 0
 Constr_Left_0_diff_aux_8: Sold_0[8] - Q0_var[8] + Left_0_aux_8 = 0
 Constr_Lost_0_diff_aux_8: - Total_Demand_Up_to_K_minus_1[8] + Q0_var[8]
   + Lost_0_aux_8 = 0
 Constr_Q1_plus_left_8: - Left_0[8] + Q1_plus_left[8] - Q1_var[8] = 0
 Constr_Left_1_diff_aux_8: Sold_1[8] - Q1_plus_left[8] + Left_0_aux_8 = 0
 Constr_Lost_1_diff_aux_8: - Total_Demand_from_k_to_T[8] + Q1_plus_left[8]
   + Lost_0_aux_8 = 0
 Profit_Constraint_8: - 700 Sold_0[8] + 700 Lost_0[8] - 700 Sold_1[8]
   + 300 Left_1[8] + 700 Lost_1[8] + profits_vars[8] = 0
 R189: 0.567546810655553 alphas[0] - 1.5004024101894 alphas[1]
   - 1.5847745563586 alphas[2] + 1.470707008847804 alphas[3] - alphas[4]
   + f_var[9] = 0
 Q0_upper_bound_9: - 2028.416054007841 Fraction_for_second_order_amount[9]
   + Q0_var[9] = 0
 tau_computation_9_0: - betas[0] + 0.567546810655553 gammas[0]
   - 1.5004024101894 gammas[1] - 1.5847745563586 gammas[2]
   + 1.470707008847804 gammas[3] + tau[9,0] = 0
 tau_computation_9_1: - betas[1] + 0.567546810655553 gammas[0]
   - 1.5004024101894 gammas[1] - 1.5847745563586 gammas[2]
   + 1.470707008847804 gammas[3] + tau[9,1] = 0
 tau_computation_9_2: - betas[2] + 0.567546810655553 gammas[0]
   - 1.5004024101894 gammas[1] - 1.5847745563586 gammas[2]
   + 1.470707008847804 gammas[3] + tau[9,2] = 0
 tau_computation_9_3: - betas[3] + 0.567546810655553 gammas[0]
   - 1.5004024101894 gammas[1] - 1.5847745563586 gammas[2]
   + 1.470707008847804 gammas[3] + tau[9,3] = 0
 tau_computation_9_4: - betas[4] + 0.567546810655553 gammas[0]
   - 1.5004024101894 gammas[1] - 1.5847745563586 gammas[2]
   + 1.470707008847804 gammas[3] + tau[9,4] = 0
 tau_computation_9_5: - betas[5] + 0.567546810655553 gammas[0]
   - 1.5004024101894 gammas[1] - 1.5847745563586 gammas[2]
   + 1.470707008847804 gammas[3] + tau[9,5] = 0
 tau_computation_9_6: - betas[6] + 0.567546810655553 gammas[0]
   - 1.5004024101894 gammas[1] - 1.5847745563586 gammas[2]
   + 1.470707008847804 gammas[3] + tau[9,6] = 0
 tau_computation_9_7: - betas[7] + 0.567546810655553 gammas[0]
   - 1.5004024101894 gammas[1] - 1.5847745563586 gammas[2]
   + 1.470707008847804 gammas[3] + tau[9,7] = 0
 Ensure_only_one_R_true_9: R[9,0] + R[9,1] + R[9,2] + R[9,3] + R[9,4]
   + R[9,5] + R[9,6] + R[9,7] = 1
 Define_Q_hat_9: - 2502.242999653194 R[9,0] - 2845.097990961776 R[9,1]
   - 2810.866782018888 R[9,2] - 2819.150738689134 R[9,3]
   - 2841.892885685933 R[9,4] - 2841.892887075263 R[9,5]
   - 2779.148690077602 R[9,6] - 2662.693067379602 R[9,7] + Q_hat[9] = 0
 Adjust_Q_hat_9: Q0_var[9] - Q_hat[9] + Q_hat_adjusted[9] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_9: Total_Demand_Up_to_K_minus_1[9]
   - 352.3589549803015 R[9,0] - 626.5267779222361 R[9,1]
   - 884.8819379307756 R[9,2] - 1159.049760975034 R[9,3]
   - 1435.32476907989 R[9,4] - 1709.492592339975 R[9,5]
   - 1983.574702889187 R[9,6] - 2114.35714324075 R[9,7] = 0
 Constr_Total_Demand_from_K_to_T_9: Total_Demand_from_k_to_T[9]
   - 2310.33383478062 R[9,0] - 2036.166011838685 R[9,1]
   - 1777.810851830145 R[9,2] - 1503.643028785887 R[9,3]
   - 1227.368020681031 R[9,4] - 953.2001974209459 R[9,5]
   - 679.1180868717342 R[9,6] - 548.3356465201711 R[9,7] = 0
 Constr_Left_0_diff_aux_9: Sold_0[9] - Q0_var[9] + Left_0_aux_9 = 0
 Constr_Lost_0_diff_aux_9: - Total_Demand_Up_to_K_minus_1[9] + Q0_var[9]
   + Lost_0_aux_9 = 0
 Constr_Q1_plus_left_9: - Left_0[9] + Q1_plus_left[9] - Q1_var[9] = 0
 Constr_Left_1_diff_aux_9: Sold_1[9] - Q1_plus_left[9] + Left_0_aux_9 = 0
 Constr_Lost_1_diff_aux_9: - Total_Demand_from_k_to_T[9] + Q1_plus_left[9]
   + Lost_0_aux_9 = 0
 Profit_Constraint_9: - 700 Sold_0[9] + 700 Lost_0[9] - 700 Sold_1[9]
   + 300 Left_1[9] + 700 Lost_1[9] + profits_vars[9] = 0
 R210: - 0.9320300178845095 alphas[0] + 0.5639561514337079 alphas[1]
   - 1.925072238254391 alphas[2] - 1.284715648791757 alphas[3] - alphas[4]
   + f_var[10] = 0
 Q0_upper_bound_10:
   - 2028.416054007841 Fraction_for_second_order_amount[10] + Q0_var[10]
   = 0
 tau_computation_10_0: - betas[0] - 0.9320300178845095 gammas[0]
   + 0.5639561514337079 gammas[1] - 1.925072238254391 gammas[2]
   - 1.284715648791757 gammas[3] + tau[10,0] = 0
 tau_computation_10_1: - betas[1] - 0.9320300178845095 gammas[0]
   + 0.5639561514337079 gammas[1] - 1.925072238254391 gammas[2]
   - 1.284715648791757 gammas[3] + tau[10,1] = 0
 tau_computation_10_2: - betas[2] - 0.9320300178845095 gammas[0]
   + 0.5639561514337079 gammas[1] - 1.925072238254391 gammas[2]
   - 1.284715648791757 gammas[3] + tau[10,2] = 0
 tau_computation_10_3: - betas[3] - 0.9320300178845095 gammas[0]
   + 0.5639561514337079 gammas[1] - 1.925072238254391 gammas[2]
   - 1.284715648791757 gammas[3] + tau[10,3] = 0
 tau_computation_10_4: - betas[4] - 0.9320300178845095 gammas[0]
   + 0.5639561514337079 gammas[1] - 1.925072238254391 gammas[2]
   - 1.284715648791757 gammas[3] + tau[10,4] = 0
 tau_computation_10_5: - betas[5] - 0.9320300178845095 gammas[0]
   + 0.5639561514337079 gammas[1] - 1.925072238254391 gammas[2]
   - 1.284715648791757 gammas[3] + tau[10,5] = 0
 tau_computation_10_6: - betas[6] - 0.9320300178845095 gammas[0]
   + 0.5639561514337079 gammas[1] - 1.925072238254391 gammas[2]
   - 1.284715648791757 gammas[3] + tau[10,6] = 0
 tau_computation_10_7: - betas[7] - 0.9320300178845095 gammas[0]
   + 0.5639561514337079 gammas[1] - 1.925072238254391 gammas[2]
   - 1.284715648791757 gammas[3] + tau[10,7] = 0
 Ensure_only_one_R_true_10: R[10,0] + R[10,1] + R[10,2] + R[10,3] + R[10,4]
   + R[10,5] + R[10,6] + R[10,7] = 1
 Define_Q_hat_10: - 1976.561219227512 R[10,0] - 2964.505902223316 R[10,1]
   - 2959.51483289229 R[10,2] - 2964.986219704103 R[10,3]
   - 3231.864873511354 R[10,4] - 3231.864805026987 R[10,5]
   - 3306.687829864194 R[10,6] - 3371.570414250383 R[10,7] + Q_hat[10] = 0
 Adjust_Q_hat_10: Q0_var[10] - Q_hat[10] + Q_hat_adjusted[10] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_10: Total_Demand_Up_to_K_minus_1[10]
   - 182.7504451341096 R[10,0] - 480.335181229569 R[10,1]
   - 745.7157284098557 R[10,2] - 1043.300464661224 R[10,3]
   - 1592.192278319523 R[10,4] - 1889.7770151687 R[10,5]
   - 2298.56091052688 R[10,6] - 2776.400662008934 R[10,7] = 0
 Constr_Total_Demand_from_K_to_T_10: Total_Demand_from_k_to_T[10]
   - 3188.819690573177 R[10,0] - 2891.234954477717 R[10,1]
   - 2625.85440729743 R[10,2] - 2328.269671046062 R[10,3]
   - 1779.377857387763 R[10,4] - 1481.793120538586 R[10,5]
   - 1073.009225180406 R[10,6] - 595.1694736983521 R[10,7] = 0
 Constr_Left_0_diff_aux_10: Sold_0[10] - Q0_var[10] + Left_0_aux_10 = 0
 Constr_Lost_0_diff_aux_10: - Total_Demand_Up_to_K_minus_1[10] + Q0_var[10]
   + Lost_0_aux_10 = 0
 Constr_Q1_plus_left_10: - Left_0[10] + Q1_plus_left[10] - Q1_var[10] = 0
 Constr_Left_1_diff_aux_10: Sold_1[10] - Q1_plus_left[10] + Left_0_aux_10
   = 0
 Constr_Lost_1_diff_aux_10: - Total_Demand_from_k_to_T[10]
   + Q1_plus_left[10] + Lost_0_aux_10 = 0
 Profit_Constraint_10: - 700 Sold_0[10] + 700 Lost_0[10] - 700 Sold_1[10]
   + 300 Left_1[10] + 700 Lost_1[10] + profits_vars[10] = 0
 R231: 0.0333137608874336 alphas[0] - 1.066236411862904 alphas[1]
   + 0.8848679582607427 alphas[2] + 0.9788265905388678 alphas[3]
   - alphas[4] + f_var[11] = 0
 Q0_upper_bound_11:
   - 2028.416054007841 Fraction_for_second_order_amount[11] + Q0_var[11]
   = 0
 tau_computation_11_0: - betas[0] + 0.0333137608874336 gammas[0]
   - 1.066236411862904 gammas[1] + 0.8848679582607427 gammas[2]
   + 0.9788265905388678 gammas[3] + tau[11,0] = 0
 tau_computation_11_1: - betas[1] + 0.0333137608874336 gammas[0]
   - 1.066236411862904 gammas[1] + 0.8848679582607427 gammas[2]
   + 0.9788265905388678 gammas[3] + tau[11,1] = 0
 tau_computation_11_2: - betas[2] + 0.0333137608874336 gammas[0]
   - 1.066236411862904 gammas[1] + 0.8848679582607427 gammas[2]
   + 0.9788265905388678 gammas[3] + tau[11,2] = 0
 tau_computation_11_3: - betas[3] + 0.0333137608874336 gammas[0]
   - 1.066236411862904 gammas[1] + 0.8848679582607427 gammas[2]
   + 0.9788265905388678 gammas[3] + tau[11,3] = 0
 tau_computation_11_4: - betas[4] + 0.0333137608874336 gammas[0]
   - 1.066236411862904 gammas[1] + 0.8848679582607427 gammas[2]
   + 0.9788265905388678 gammas[3] + tau[11,4] = 0
 tau_computation_11_5: - betas[5] + 0.0333137608874336 gammas[0]
   - 1.066236411862904 gammas[1] + 0.8848679582607427 gammas[2]
   + 0.9788265905388678 gammas[3] + tau[11,5] = 0
 tau_computation_11_6: - betas[6] + 0.0333137608874336 gammas[0]
   - 1.066236411862904 gammas[1] + 0.8848679582607427 gammas[2]
   + 0.9788265905388678 gammas[3] + tau[11,6] = 0
 tau_computation_11_7: - betas[7] + 0.0333137608874336 gammas[0]
   - 1.066236411862904 gammas[1] + 0.8848679582607427 gammas[2]
   + 0.9788265905388678 gammas[3] + tau[11,7] = 0
 Ensure_only_one_R_true_11: R[11,0] + R[11,1] + R[11,2] + R[11,3] + R[11,4]
   + R[11,5] + R[11,6] + R[11,7] = 1
 Define_Q_hat_11: - 1628.220289883084 R[11,0] - 1138.705921811456 R[11,1]
   - 1081.358832993585 R[11,2] - 1041.347931701299 R[11,3]
   - 1318.333047668124 R[11,4] - 1318.333063177711 R[11,5]
   - 1386.790444893325 R[11,6] - 1294.229461207121 R[11,7] + Q_hat[11] = 0
 Adjust_Q_hat_11: Q0_var[11] - Q_hat[11] + Q_hat_adjusted[11] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_11: Total_Demand_Up_to_K_minus_1[11]
   - 70.36004443065846 R[11,0] - 174.4208702673298 R[11,1]
   - 174.4208702673298 R[11,2] - 278.6451158342506 R[11,3]
   - 636.8887494028797 R[11,4] - 741.1129955024983 R[11,5]
   - 1013.751976992206 R[11,6] - 1085.780691702743 R[11,7] = 0
 Constr_Total_Demand_from_K_to_T_11: Total_Demand_from_k_to_T[11]
   - 1223.869139471219 R[11,0] - 1119.808313634547 R[11,1]
   - 1119.808313634547 R[11,2] - 1015.584068067626 R[11,3]
   - 657.3404344989972 R[11,4] - 553.1161883993786 R[11,5]
   - 280.4772069096706 R[11,6] - 208.4484921991337 R[11,7] = 0
 Constr_Left_0_diff_aux_11: Sold_0[11] - Q0_var[11] + Left_0_aux_11 = 0
 Constr_Lost_0_diff_aux_11: - Total_Demand_Up_to_K_minus_1[11] + Q0_var[11]
   + Lost_0_aux_11 = 0
 Constr_Q1_plus_left_11: - Left_0[11] + Q1_plus_left[11] - Q1_var[11] = 0
 Constr_Left_1_diff_aux_11: Sold_1[11] - Q1_plus_left[11] + Left_0_aux_11
   = 0
 Constr_Lost_1_diff_aux_11: - Total_Demand_from_k_to_T[11]
   + Q1_plus_left[11] + Lost_0_aux_11 = 0
 Profit_Constraint_11: - 700 Sold_0[11] + 700 Lost_0[11] - 700 Sold_1[11]
   + 300 Left_1[11] + 700 Lost_1[11] + profits_vars[11] = 0
 R252: - 0.1104782175581155 alphas[0] - 0.4663535360785392 alphas[1]
   - 0.7361231455833085 alphas[2] + 1.223550772776169 alphas[3] - alphas[4]
   + f_var[12] = 0
 Q0_upper_bound_12:
   - 2028.416054007841 Fraction_for_second_order_amount[12] + Q0_var[12]
   = 0
 tau_computation_12_0: - betas[0] - 0.1104782175581155 gammas[0]
   - 0.4663535360785392 gammas[1] - 0.7361231455833085 gammas[2]
   + 1.223550772776169 gammas[3] + tau[12,0] = 0
 tau_computation_12_1: - betas[1] - 0.1104782175581155 gammas[0]
   - 0.4663535360785392 gammas[1] - 0.7361231455833085 gammas[2]
   + 1.223550772776169 gammas[3] + tau[12,1] = 0
 tau_computation_12_2: - betas[2] - 0.1104782175581155 gammas[0]
   - 0.4663535360785392 gammas[1] - 0.7361231455833085 gammas[2]
   + 1.223550772776169 gammas[3] + tau[12,2] = 0
 tau_computation_12_3: - betas[3] - 0.1104782175581155 gammas[0]
   - 0.4663535360785392 gammas[1] - 0.7361231455833085 gammas[2]
   + 1.223550772776169 gammas[3] + tau[12,3] = 0
 tau_computation_12_4: - betas[4] - 0.1104782175581155 gammas[0]
   - 0.4663535360785392 gammas[1] - 0.7361231455833085 gammas[2]
   + 1.223550772776169 gammas[3] + tau[12,4] = 0
 tau_computation_12_5: - betas[5] - 0.1104782175581155 gammas[0]
   - 0.4663535360785392 gammas[1] - 0.7361231455833085 gammas[2]
   + 1.223550772776169 gammas[3] + tau[12,5] = 0
 tau_computation_12_6: - betas[6] - 0.1104782175581155 gammas[0]
   - 0.4663535360785392 gammas[1] - 0.7361231455833085 gammas[2]
   + 1.223550772776169 gammas[3] + tau[12,6] = 0
 tau_computation_12_7: - betas[7] - 0.1104782175581155 gammas[0]
   - 0.4663535360785392 gammas[1] - 0.7361231455833085 gammas[2]
   + 1.223550772776169 gammas[3] + tau[12,7] = 0
 Ensure_only_one_R_true_12: R[12,0] + R[12,1] + R[12,2] + R[12,3] + R[12,4]
   + R[12,5] + R[12,6] + R[12,7] = 1
 Define_Q_hat_12: - 1867.458313186437 R[12,0] - 2199.444094516891 R[12,1]
   - 2221.16761748128 R[12,2] - 2215.645499171367 R[12,3]
   - 2361.103358512666 R[12,4] - 2361.103312591373 R[12,5]
   - 2271.958459596659 R[12,6] - 2234.667275303906 R[12,7] + Q_hat[12] = 0
 Adjust_Q_hat_12: Q0_var[12] - Q_hat[12] + Q_hat_adjusted[12] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_12: Total_Demand_Up_to_K_minus_1[12]
   - 147.5489563198478 R[12,0] - 363.3185059562825 R[12,1]
   - 619.177735658586 R[12,2] - 834.9472862594378 R[12,3]
   - 1179.784880639054 R[12,4] - 1395.554431414079 R[12,5]
   - 1545.962471891993 R[12,6] - 1803.127894967017 R[12,7] = 0
 Constr_Total_Demand_from_K_to_T_12: Total_Demand_from_k_to_T[12]
   - 2087.11804019722 R[12,0] - 1871.348490560785 R[12,1]
   - 1615.489260858481 R[12,2] - 1399.71971025763 R[12,3]
   - 1054.882115878014 R[12,4] - 839.1125651029884 R[12,5]
   - 688.7045246250743 R[12,6] - 431.5391015500504 R[12,7] = 0
 Constr_Left_0_diff_aux_12: Sold_0[12] - Q0_var[12] + Left_0_aux_12 = 0
 Constr_Lost_0_diff_aux_12: - Total_Demand_Up_to_K_minus_1[12] + Q0_var[12]
   + Lost_0_aux_12 = 0
 Constr_Q1_plus_left_12: - Left_0[12] + Q1_plus_left[12] - Q1_var[12] = 0
 Constr_Left_1_diff_aux_12: Sold_1[12] - Q1_plus_left[12] + Left_0_aux_12
   = 0
 Constr_Lost_1_diff_aux_12: - Total_Demand_from_k_to_T[12]
   + Q1_plus_left[12] + Lost_0_aux_12 = 0
 Profit_Constraint_12: - 700 Sold_0[12] + 700 Lost_0[12] - 700 Sold_1[12]
   + 300 Left_1[12] + 700 Lost_1[12] + profits_vars[12] = 0
 R273: - 1.423724490489904 alphas[0] + 1.175103588620279 alphas[1]
   + 0.7162029745743949 alphas[2] + 1.186960952297952 alphas[3] - alphas[4]
   + f_var[13] = 0
 Q0_upper_bound_13:
   - 2028.416054007841 Fraction_for_second_order_amount[13] + Q0_var[13]
   = 0
 tau_computation_13_0: - betas[0] - 1.423724490489904 gammas[0]
   + 1.175103588620279 gammas[1] + 0.7162029745743949 gammas[2]
   + 1.186960952297952 gammas[3] + tau[13,0] = 0
 tau_computation_13_1: - betas[1] - 1.423724490489904 gammas[0]
   + 1.175103588620279 gammas[1] + 0.7162029745743949 gammas[2]
   + 1.186960952297952 gammas[3] + tau[13,1] = 0
 tau_computation_13_2: - betas[2] - 1.423724490489904 gammas[0]
   + 1.175103588620279 gammas[1] + 0.7162029745743949 gammas[2]
   + 1.186960952297952 gammas[3] + tau[13,2] = 0
 tau_computation_13_3: - betas[3] - 1.423724490489904 gammas[0]
   + 1.175103588620279 gammas[1] + 0.7162029745743949 gammas[2]
   + 1.186960952297952 gammas[3] + tau[13,3] = 0
 tau_computation_13_4: - betas[4] - 1.423724490489904 gammas[0]
   + 1.175103588620279 gammas[1] + 0.7162029745743949 gammas[2]
   + 1.186960952297952 gammas[3] + tau[13,4] = 0
 tau_computation_13_5: - betas[5] - 1.423724490489904 gammas[0]
   + 1.175103588620279 gammas[1] + 0.7162029745743949 gammas[2]
   + 1.186960952297952 gammas[3] + tau[13,5] = 0
 tau_computation_13_6: - betas[6] - 1.423724490489904 gammas[0]
   + 1.175103588620279 gammas[1] + 0.7162029745743949 gammas[2]
   + 1.186960952297952 gammas[3] + tau[13,6] = 0
 tau_computation_13_7: - betas[7] - 1.423724490489904 gammas[0]
   + 1.175103588620279 gammas[1] + 0.7162029745743949 gammas[2]
   + 1.186960952297952 gammas[3] + tau[13,7] = 0
 Ensure_only_one_R_true_13: R[13,0] + R[13,1] + R[13,2] + R[13,3] + R[13,4]
   + R[13,5] + R[13,6] + R[13,7] = 1
 Define_Q_hat_13: - 2059.013107476195 R[13,0] - 1322.981942283153 R[13,1]
   - 1333.349233419108 R[13,2] - 1322.445455724223 R[13,3]
   - 1303.404940437687 R[13,4] - 1303.404886836354 R[13,5]
   - 1350.307515950904 R[13,6] - 1258.306820757196 R[13,7] + Q_hat[13] = 0
 Adjust_Q_hat_13: Q0_var[13] - Q_hat[13] + Q_hat_adjusted[13] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_13: Total_Demand_Up_to_K_minus_1[13]
   - 209.353120784271 R[13,0] - 325.1840798913786 R[13,1]
   - 521.8631166418313 R[13,2] - 637.6937106718317 R[13,3]
   - 701.6717037030804 R[13,4] - 817.5022978873681 R[13,5]
   - 961.1051724135359 R[13,6] - 1026.645354231385 R[13,7] = 0
 Constr_Total_Demand_from_K_to_T_13: Total_Demand_from_k_to_T[13]
   - 1048.953421815689 R[13,0] - 933.1224627085817 R[13,1]
   - 736.4434259581288 R[13,2] - 620.6128319281285 R[13,3]
   - 556.6348388968797 R[13,4] - 440.8042447125921 R[13,5]
   - 297.2013701864244 R[13,6] - 231.6611883685753 R[13,7] = 0
 Constr_Left_0_diff_aux_13: Sold_0[13] - Q0_var[13] + Left_0_aux_13 = 0
 Constr_Lost_0_diff_aux_13: - Total_Demand_Up_to_K_minus_1[13] + Q0_var[13]
   + Lost_0_aux_13 = 0
 Constr_Q1_plus_left_13: - Left_0[13] + Q1_plus_left[13] - Q1_var[13] = 0
 Constr_Left_1_diff_aux_13: Sold_1[13] - Q1_plus_left[13] + Left_0_aux_13
   = 0
 Constr_Lost_1_diff_aux_13: - Total_Demand_from_k_to_T[13]
   + Q1_plus_left[13] + Lost_0_aux_13 = 0
 Profit_Constraint_13: - 700 Sold_0[13] + 700 Lost_0[13] - 700 Sold_1[13]
   + 300 Left_1[13] + 700 Lost_1[13] + profits_vars[13] = 0
 R294: 1.714974928057006 alphas[0] - 0.9330929258103161 alphas[1]
   + 1.597836103058035 alphas[2] - 0.9270805016014423 alphas[3] - alphas[4]
   + f_var[14] = 0
 Q0_upper_bound_14:
   - 2028.416054007841 Fraction_for_second_order_amount[14] + Q0_var[14]
   = 0
 tau_computation_14_0: - betas[0] + 1.714974928057006 gammas[0]
   - 0.9330929258103161 gammas[1] + 1.597836103058035 gammas[2]
   - 0.9270805016014423 gammas[3] + tau[14,0] = 0
 tau_computation_14_1: - betas[1] + 1.714974928057006 gammas[0]
   - 0.9330929258103161 gammas[1] + 1.597836103058035 gammas[2]
   - 0.9270805016014423 gammas[3] + tau[14,1] = 0
 tau_computation_14_2: - betas[2] + 1.714974928057006 gammas[0]
   - 0.9330929258103161 gammas[1] + 1.597836103058035 gammas[2]
   - 0.9270805016014423 gammas[3] + tau[14,2] = 0
 tau_computation_14_3: - betas[3] + 1.714974928057006 gammas[0]
   - 0.9330929258103161 gammas[1] + 1.597836103058035 gammas[2]
   - 0.9270805016014423 gammas[3] + tau[14,3] = 0
 tau_computation_14_4: - betas[4] + 1.714974928057006 gammas[0]
   - 0.9330929258103161 gammas[1] + 1.597836103058035 gammas[2]
   - 0.9270805016014423 gammas[3] + tau[14,4] = 0
 tau_computation_14_5: - betas[5] + 1.714974928057006 gammas[0]
   - 0.9330929258103161 gammas[1] + 1.597836103058035 gammas[2]
   - 0.9270805016014423 gammas[3] + tau[14,5] = 0
 tau_computation_14_6: - betas[6] + 1.714974928057006 gammas[0]
   - 0.9330929258103161 gammas[1] + 1.597836103058035 gammas[2]
   - 0.9270805016014423 gammas[3] + tau[14,6] = 0
 tau_computation_14_7: - betas[7] + 1.714974928057006 gammas[0]
   - 0.9330929258103161 gammas[1] + 1.597836103058035 gammas[2]
   - 0.9270805016014423 gammas[3] + tau[14,7] = 0
 Ensure_only_one_R_true_14: R[14,0] + R[14,1] + R[14,2] + R[14,3] + R[14,4]
   + R[14,5] + R[14,6] + R[14,7] = 1
 Define_Q_hat_14: - 2138.283508597003 R[14,0] - 780.5911746807969 R[14,1]
   - 749.893442651462 R[14,2] - 635.9479659638964 R[14,3]
   - 665.9758665409399 R[14,4] - 665.9761692891279 R[14,5]
   - 589.1376548283941 R[14,6] - 597.7311343591938 R[14,7] + Q_hat[14] = 0
 Adjust_Q_hat_14: Q0_var[14] - Q_hat[14] + Q_hat_adjusted[14] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_14: Total_Demand_Up_to_K_minus_1[14]
   - 234.9293060538517 R[14,0] - 289.6846367509716 R[14,1]
   - 377.0798588775787 R[14,2] - 432.242609292058 R[14,3]
   - 432.242609292058 R[14,4] - 487.4053591546971 R[14,5]
   - 487.4053591546971 R[14,6] - 487.4053591546971 R[14,7] = 0
 Constr_Total_Demand_from_K_to_T_14: Total_Demand_from_k_to_T[14]
   - 362.8015522349154 R[14,0] - 308.0462215377955 R[14,1]
   - 220.6509994111885 R[14,2] - 165.4882489967092 R[14,3]
   - 165.4882489967092 R[14,4] - 110.32549913407 R[14,5]
   - 110.32549913407 R[14,6] - 110.32549913407 R[14,7] = 0
 Constr_Left_0_diff_aux_14: Sold_0[14] - Q0_var[14] + Left_0_aux_14 = 0
 Constr_Lost_0_diff_aux_14: - Total_Demand_Up_to_K_minus_1[14] + Q0_var[14]
   + Lost_0_aux_14 = 0
 Constr_Q1_plus_left_14: - Left_0[14] + Q1_plus_left[14] - Q1_var[14] = 0
 Constr_Left_1_diff_aux_14: Sold_1[14] - Q1_plus_left[14] + Left_0_aux_14
   = 0
 Constr_Lost_1_diff_aux_14: - Total_Demand_from_k_to_T[14]
   + Q1_plus_left[14] + Lost_0_aux_14 = 0
 Profit_Constraint_14: - 700 Sold_0[14] + 700 Lost_0[14] - 700 Sold_1[14]
   + 300 Left_1[14] + 700 Lost_1[14] + profits_vars[14] = 0
 R315: 1.655866365215913 alphas[0] + 0.7706258476638413 alphas[1]
   - 1.08235283065384 alphas[2] + 0.6215993257875255 alphas[3] - alphas[4]
   + f_var[15] = 0
 Q0_upper_bound_15:
   - 2028.416054007841 Fraction_for_second_order_amount[15] + Q0_var[15]
   = 0
 tau_computation_15_0: - betas[0] + 1.655866365215913 gammas[0]
   + 0.7706258476638413 gammas[1] - 1.08235283065384 gammas[2]
   + 0.6215993257875255 gammas[3] + tau[15,0] = 0
 tau_computation_15_1: - betas[1] + 1.655866365215913 gammas[0]
   + 0.7706258476638413 gammas[1] - 1.08235283065384 gammas[2]
   + 0.6215993257875255 gammas[3] + tau[15,1] = 0
 tau_computation_15_2: - betas[2] + 1.655866365215913 gammas[0]
   + 0.7706258476638413 gammas[1] - 1.08235283065384 gammas[2]
   + 0.6215993257875255 gammas[3] + tau[15,2] = 0
 tau_computation_15_3: - betas[3] + 1.655866365215913 gammas[0]
   + 0.7706258476638413 gammas[1] - 1.08235283065384 gammas[2]
   + 0.6215993257875255 gammas[3] + tau[15,3] = 0
 tau_computation_15_4: - betas[4] + 1.655866365215913 gammas[0]
   + 0.7706258476638413 gammas[1] - 1.08235283065384 gammas[2]
   + 0.6215993257875255 gammas[3] + tau[15,4] = 0
 tau_computation_15_5: - betas[5] + 1.655866365215913 gammas[0]
   + 0.7706258476638413 gammas[1] - 1.08235283065384 gammas[2]
   + 0.6215993257875255 gammas[3] + tau[15,5] = 0
 tau_computation_15_6: - betas[6] + 1.655866365215913 gammas[0]
   + 0.7706258476638413 gammas[1] - 1.08235283065384 gammas[2]
   + 0.6215993257875255 gammas[3] + tau[15,6] = 0
 tau_computation_15_7: - betas[7] + 1.655866365215913 gammas[0]
   + 0.7706258476638413 gammas[1] - 1.08235283065384 gammas[2]
   + 0.6215993257875255 gammas[3] + tau[15,7] = 0
 Ensure_only_one_R_true_15: R[15,0] + R[15,1] + R[15,2] + R[15,3] + R[15,4]
   + R[15,5] + R[15,6] + R[15,7] = 1
 Define_Q_hat_15: - 2394.158400275291 R[15,0] - 2510.719963969389 R[15,1]
   - 2461.133904098697 R[15,2] - 2470.232432617241 R[15,3]
   - 2373.628564082903 R[15,4] - 2373.628644308445 R[15,5]
   - 2489.182993379662 R[15,6] - 2409.019149629327 R[15,7] + Q_hat[15] = 0
 Adjust_Q_hat_15: Q0_var[15] - Q_hat[15] + Q_hat_adjusted[15] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_15: Total_Demand_Up_to_K_minus_1[15]
   - 317.4860175485267 R[15,0] - 557.0806810756881 R[15,1]
   - 750.2649693970222 R[15,2] - 989.8596332465289 R[15,3]
   - 1152.007690235898 R[15,4] - 1391.602353694933 R[15,5]
   - 1775.372501956559 R[15,6] - 1929.829545184692 R[15,7] = 0
 Constr_Total_Demand_from_K_to_T_15: Total_Demand_from_k_to_T[15]
   - 2091.532854554236 R[15,0] - 1851.938191027074 R[15,1]
   - 1658.75390270574 R[15,2] - 1419.159238856234 R[15,3]
   - 1257.011181866865 R[15,4] - 1017.416518407829 R[15,5]
   - 633.646370146203 R[15,6] - 479.1893269180707 R[15,7] = 0
 Constr_Left_0_diff_aux_15: Sold_0[15] - Q0_var[15] + Left_0_aux_15 = 0
 Constr_Lost_0_diff_aux_15: - Total_Demand_Up_to_K_minus_1[15] + Q0_var[15]
   + Lost_0_aux_15 = 0
 Constr_Q1_plus_left_15: - Left_0[15] + Q1_plus_left[15] - Q1_var[15] = 0
 Constr_Left_1_diff_aux_15: Sold_1[15] - Q1_plus_left[15] + Left_0_aux_15
   = 0
 Constr_Lost_1_diff_aux_15: - Total_Demand_from_k_to_T[15]
   + Q1_plus_left[15] + Lost_0_aux_15 = 0
 Profit_Constraint_15: - 700 Sold_0[15] + 700 Lost_0[15] - 700 Sold_1[15]
   + 300 Left_1[15] + 700 Lost_1[15] + profits_vars[15] = 0
 R336: 1.901622154698307 alphas[0] - 1.207887106810452 alphas[1]
   + 0.5102289878293261 alphas[2] - 0.0142058691414214 alphas[3]
   - alphas[4] + f_var[16] = 0
 Q0_upper_bound_16:
   - 2028.416054007841 Fraction_for_second_order_amount[16] + Q0_var[16]
   = 0
 tau_computation_16_0: - betas[0] + 1.901622154698307 gammas[0]
   - 1.207887106810452 gammas[1] + 0.5102289878293261 gammas[2]
   - 0.0142058691414214 gammas[3] + tau[16,0] = 0
 tau_computation_16_1: - betas[1] + 1.901622154698307 gammas[0]
   - 1.207887106810452 gammas[1] + 0.5102289878293261 gammas[2]
   - 0.0142058691414214 gammas[3] + tau[16,1] = 0
 tau_computation_16_2: - betas[2] + 1.901622154698307 gammas[0]
   - 1.207887106810452 gammas[1] + 0.5102289878293261 gammas[2]
   - 0.0142058691414214 gammas[3] + tau[16,2] = 0
 tau_computation_16_3: - betas[3] + 1.901622154698307 gammas[0]
   - 1.207887106810452 gammas[1] + 0.5102289878293261 gammas[2]
   - 0.0142058691414214 gammas[3] + tau[16,3] = 0
 tau_computation_16_4: - betas[4] + 1.901622154698307 gammas[0]
   - 1.207887106810452 gammas[1] + 0.5102289878293261 gammas[2]
   - 0.0142058691414214 gammas[3] + tau[16,4] = 0
 tau_computation_16_5: - betas[5] + 1.901622154698307 gammas[0]
   - 1.207887106810452 gammas[1] + 0.5102289878293261 gammas[2]
   - 0.0142058691414214 gammas[3] + tau[16,5] = 0
 tau_computation_16_6: - betas[6] + 1.901622154698307 gammas[0]
   - 1.207887106810452 gammas[1] + 0.5102289878293261 gammas[2]
   - 0.0142058691414214 gammas[3] + tau[16,6] = 0
 tau_computation_16_7: - betas[7] + 1.901622154698307 gammas[0]
   - 1.207887106810452 gammas[1] + 0.5102289878293261 gammas[2]
   - 0.0142058691414214 gammas[3] + tau[16,7] = 0
 Ensure_only_one_R_true_16: R[16,0] + R[16,1] + R[16,2] + R[16,3] + R[16,4]
   + R[16,5] + R[16,6] + R[16,7] = 1
 Define_Q_hat_16: - 1843.883798917036 R[16,0] - 1413.646296993337 R[16,1]
   - 1397.208930397738 R[16,2] - 1392.216943362569 R[16,3]
   - 1267.153739638272 R[16,4] - 1267.153791156626 R[16,5]
   - 1302.287395040094 R[16,6] - 1137.676860756748 R[16,7] + Q_hat[16] = 0
 Adjust_Q_hat_16: Q0_var[16] - Q_hat[16] + Q_hat_adjusted[16] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_16: Total_Demand_Up_to_K_minus_1[16]
   - 139.9427611187452 R[16,0] - 269.9436216078774 R[16,1]
   - 394.5653788356331 R[16,2] - 524.5696668918995 R[16,3]
   - 571.930271830922 R[16,4] - 701.934559536539 R[16,5]
   - 877.6680076190777 R[16,6] - 877.6680076190777 R[16,7] = 0
 Constr_Total_Demand_from_K_to_T_16: Total_Demand_from_k_to_T[16]
   - 997.7338219115807 R[16,0] - 867.7329614224484 R[16,1]
   - 743.1112041946928 R[16,2] - 613.1069161384263 R[16,3]
   - 565.7463111994037 R[16,4] - 435.7420234937867 R[16,5]
   - 260.0085754112481 R[16,6] - 260.0085754112481 R[16,7] = 0
 Constr_Left_0_diff_aux_16: Sold_0[16] - Q0_var[16] + Left_0_aux_16 = 0
 Constr_Lost_0_diff_aux_16: - Total_Demand_Up_to_K_minus_1[16] + Q0_var[16]
   + Lost_0_aux_16 = 0
 Constr_Q1_plus_left_16: - Left_0[16] + Q1_plus_left[16] - Q1_var[16] = 0
 Constr_Left_1_diff_aux_16: Sold_1[16] - Q1_plus_left[16] + Left_0_aux_16
   = 0
 Constr_Lost_1_diff_aux_16: - Total_Demand_from_k_to_T[16]
   + Q1_plus_left[16] + Lost_0_aux_16 = 0
 Profit_Constraint_16: - 700 Sold_0[16] + 700 Lost_0[16] - 700 Sold_1[16]
   + 300 Left_1[16] + 700 Lost_1[16] + profits_vars[16] = 0
 R357: - 1.082231784035022 alphas[0] + 0.0162591891519979 alphas[1]
   + 0.2797248737463011 alphas[2] - 0.5108415636449818 alphas[3]
   - alphas[4] + f_var[17] = 0
 Q0_upper_bound_17:
   - 2028.416054007841 Fraction_for_second_order_amount[17] + Q0_var[17]
   = 0
 tau_computation_17_0: - betas[0] - 1.082231784035022 gammas[0]
   + 0.0162591891519979 gammas[1] + 0.2797248737463011 gammas[2]
   - 0.5108415636449818 gammas[3] + tau[17,0] = 0
 tau_computation_17_1: - betas[1] - 1.082231784035022 gammas[0]
   + 0.0162591891519979 gammas[1] + 0.2797248737463011 gammas[2]
   - 0.5108415636449818 gammas[3] + tau[17,1] = 0
 tau_computation_17_2: - betas[2] - 1.082231784035022 gammas[0]
   + 0.0162591891519979 gammas[1] + 0.2797248737463011 gammas[2]
   - 0.5108415636449818 gammas[3] + tau[17,2] = 0
 tau_computation_17_3: - betas[3] - 1.082231784035022 gammas[0]
   + 0.0162591891519979 gammas[1] + 0.2797248737463011 gammas[2]
   - 0.5108415636449818 gammas[3] + tau[17,3] = 0
 tau_computation_17_4: - betas[4] - 1.082231784035022 gammas[0]
   + 0.0162591891519979 gammas[1] + 0.2797248737463011 gammas[2]
   - 0.5108415636449818 gammas[3] + tau[17,4] = 0
 tau_computation_17_5: - betas[5] - 1.082231784035022 gammas[0]
   + 0.0162591891519979 gammas[1] + 0.2797248737463011 gammas[2]
   - 0.5108415636449818 gammas[3] + tau[17,5] = 0
 tau_computation_17_6: - betas[6] - 1.082231784035022 gammas[0]
   + 0.0162591891519979 gammas[1] + 0.2797248737463011 gammas[2]
   - 0.5108415636449818 gammas[3] + tau[17,6] = 0
 tau_computation_17_7: - betas[7] - 1.082231784035022 gammas[0]
   + 0.0162591891519979 gammas[1] + 0.2797248737463011 gammas[2]
   - 0.5108415636449818 gammas[3] + tau[17,7] = 0
 Ensure_only_one_R_true_17: R[17,0] + R[17,1] + R[17,2] + R[17,3] + R[17,4]
   + R[17,5] + R[17,6] + R[17,7] = 1
 Define_Q_hat_17: - 1423.491891471529 R[17,0] - 1483.050080535403 R[17,1]
   - 1459.335788413979 R[17,2] - 1458.572514943279 R[17,3]
   - 1364.926719160393 R[17,4] - 1364.92675073659 R[17,5]
   - 1393.39743978607 R[17,6] - 1292.121996752742 R[17,7] + Q_hat[17] = 0
 Adjust_Q_hat_17: Q0_var[17] - Q_hat[17] + Q_hat_adjusted[17] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_17: Total_Demand_Up_to_K_minus_1[17]
   - 4.305484674634272 R[17,0] - 150.1702504457939 R[17,1]
   - 217.2323815097403 R[17,2] - 363.0983548486594 R[17,3]
   - 517.9158842164272 R[17,4] - 663.7818577594018 R[17,5]
   - 874.8155159711745 R[17,6] - 1000.389772420226 R[17,7] = 0
 Constr_Total_Demand_from_K_to_T_17: Total_Demand_from_k_to_T[17]
   - 1287.816234831541 R[17,0] - 1141.951469060381 R[17,1]
   - 1074.889337996435 R[17,2] - 929.0233646575159 R[17,3]
   - 774.2058352897482 R[17,4] - 628.3398617467737 R[17,5]
   - 417.3062035350009 R[17,6] - 291.731947085949 R[17,7] = 0
 Constr_Left_0_diff_aux_17: Sold_0[17] - Q0_var[17] + Left_0_aux_17 = 0
 Constr_Lost_0_diff_aux_17: - Total_Demand_Up_to_K_minus_1[17] + Q0_var[17]
   + Lost_0_aux_17 = 0
 Constr_Q1_plus_left_17: - Left_0[17] + Q1_plus_left[17] - Q1_var[17] = 0
 Constr_Left_1_diff_aux_17: Sold_1[17] - Q1_plus_left[17] + Left_0_aux_17
   = 0
 Constr_Lost_1_diff_aux_17: - Total_Demand_from_k_to_T[17]
   + Q1_plus_left[17] + Lost_0_aux_17 = 0
 Profit_Constraint_17: - 700 Sold_0[17] + 700 Lost_0[17] - 700 Sold_1[17]
   + 300 Left_1[17] + 700 Lost_1[17] + profits_vars[17] = 0
 R378: - 0.8821953122750865 alphas[0] - 1.210468107017091 alphas[1]
   - 0.4645136380804882 alphas[2] - 1.611272289942189 alphas[3] - alphas[4]
   + f_var[18] = 0
 Q0_upper_bound_18:
   - 2028.416054007841 Fraction_for_second_order_amount[18] + Q0_var[18]
   = 0
 tau_computation_18_0: - betas[0] - 0.8821953122750865 gammas[0]
   - 1.210468107017091 gammas[1] - 0.4645136380804882 gammas[2]
   - 1.611272289942189 gammas[3] + tau[18,0] = 0
 tau_computation_18_1: - betas[1] - 0.8821953122750865 gammas[0]
   - 1.210468107017091 gammas[1] - 0.4645136380804882 gammas[2]
   - 1.611272289942189 gammas[3] + tau[18,1] = 0
 tau_computation_18_2: - betas[2] - 0.8821953122750865 gammas[0]
   - 1.210468107017091 gammas[1] - 0.4645136380804882 gammas[2]
   - 1.611272289942189 gammas[3] + tau[18,2] = 0
 tau_computation_18_3: - betas[3] - 0.8821953122750865 gammas[0]
   - 1.210468107017091 gammas[1] - 0.4645136380804882 gammas[2]
   - 1.611272289942189 gammas[3] + tau[18,3] = 0
 tau_computation_18_4: - betas[4] - 0.8821953122750865 gammas[0]
   - 1.210468107017091 gammas[1] - 0.4645136380804882 gammas[2]
   - 1.611272289942189 gammas[3] + tau[18,4] = 0
 tau_computation_18_5: - betas[5] - 0.8821953122750865 gammas[0]
   - 1.210468107017091 gammas[1] - 0.4645136380804882 gammas[2]
   - 1.611272289942189 gammas[3] + tau[18,5] = 0
 tau_computation_18_6: - betas[6] - 0.8821953122750865 gammas[0]
   - 1.210468107017091 gammas[1] - 0.4645136380804882 gammas[2]
   - 1.611272289942189 gammas[3] + tau[18,6] = 0
 tau_computation_18_7: - betas[7] - 0.8821953122750865 gammas[0]
   - 1.210468107017091 gammas[1] - 0.4645136380804882 gammas[2]
   - 1.611272289942189 gammas[3] + tau[18,7] = 0
 Ensure_only_one_R_true_18: R[18,0] + R[18,1] + R[18,2] + R[18,3] + R[18,4]
   + R[18,5] + R[18,6] + R[18,7] = 1
 Define_Q_hat_18: - 2014.566037971012 R[18,0] - 2055.426767625727 R[18,1]
   - 2067.624929083133 R[18,2] - 2062.299849951099 R[18,3]
   - 1897.892525533345 R[18,4] - 1897.89259983234 R[18,5]
   - 1904.945804921348 R[18,6] - 2057.826148612096 R[18,7] + Q_hat[18] = 0
 Adjust_Q_hat_18: Q0_var[18] - Q_hat[18] + Q_hat_adjusted[18] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_18: Total_Demand_Up_to_K_minus_1[18]
   - 195.0125035011314 R[18,0] - 392.0918750905658 R[18,1]
   - 635.962439916675 R[18,2] - 833.041718717741 R[18,3]
   - 902.4263783058246 R[18,4] - 1099.505656694226 R[18,5]
   - 1272.290010689036 R[18,6] - 1663.667313796551 R[18,7] = 0
 Constr_Total_Demand_from_K_to_T_18: Total_Demand_from_k_to_T[18]
   - 1862.813367072223 R[18,0] - 1665.733995482788 R[18,1]
   - 1421.863430656679 R[18,2] - 1224.784151855613 R[18,3]
   - 1155.39949226753 R[18,4] - 958.3202138791282 R[18,5]
   - 785.535859884318 R[18,6] - 394.1585567768029 R[18,7] = 0
 Constr_Left_0_diff_aux_18: Sold_0[18] - Q0_var[18] + Left_0_aux_18 = 0
 Constr_Lost_0_diff_aux_18: - Total_Demand_Up_to_K_minus_1[18] + Q0_var[18]
   + Lost_0_aux_18 = 0
 Constr_Q1_plus_left_18: - Left_0[18] + Q1_plus_left[18] - Q1_var[18] = 0
 Constr_Left_1_diff_aux_18: Sold_1[18] - Q1_plus_left[18] + Left_0_aux_18
   = 0
 Constr_Lost_1_diff_aux_18: - Total_Demand_from_k_to_T[18]
   + Q1_plus_left[18] + Lost_0_aux_18 = 0
 Profit_Constraint_18: - 700 Sold_0[18] + 700 Lost_0[18] - 700 Sold_1[18]
   + 300 Left_1[18] + 700 Lost_1[18] + profits_vars[18] = 0
 R399: - 1.219569266564456 alphas[0] - 0.4379875488601938 alphas[1]
   - 1.346364619250364 alphas[2] - 1.126695627034258 alphas[3] - alphas[4]
   + f_var[19] = 0
 Q0_upper_bound_19:
   - 2028.416054007841 Fraction_for_second_order_amount[19] + Q0_var[19]
   = 0
 tau_computation_19_0: - betas[0] - 1.219569266564456 gammas[0]
   - 0.4379875488601938 gammas[1] - 1.346364619250364 gammas[2]
   - 1.126695627034258 gammas[3] + tau[19,0] = 0
 tau_computation_19_1: - betas[1] - 1.219569266564456 gammas[0]
   - 0.4379875488601938 gammas[1] - 1.346364619250364 gammas[2]
   - 1.126695627034258 gammas[3] + tau[19,1] = 0
 tau_computation_19_2: - betas[2] - 1.219569266564456 gammas[0]
   - 0.4379875488601938 gammas[1] - 1.346364619250364 gammas[2]
   - 1.126695627034258 gammas[3] + tau[19,2] = 0
 tau_computation_19_3: - betas[3] - 1.219569266564456 gammas[0]
   - 0.4379875488601938 gammas[1] - 1.346364619250364 gammas[2]
   - 1.126695627034258 gammas[3] + tau[19,3] = 0
 tau_computation_19_4: - betas[4] - 1.219569266564456 gammas[0]
   - 0.4379875488601938 gammas[1] - 1.346364619250364 gammas[2]
   - 1.126695627034258 gammas[3] + tau[19,4] = 0
 tau_computation_19_5: - betas[5] - 1.219569266564456 gammas[0]
   - 0.4379875488601938 gammas[1] - 1.346364619250364 gammas[2]
   - 1.126695627034258 gammas[3] + tau[19,5] = 0
 tau_computation_19_6: - betas[6] - 1.219569266564456 gammas[0]
   - 0.4379875488601938 gammas[1] - 1.346364619250364 gammas[2]
   - 1.126695627034258 gammas[3] + tau[19,6] = 0
 tau_computation_19_7: - betas[7] - 1.219569266564456 gammas[0]
   - 0.4379875488601938 gammas[1] - 1.346364619250364 gammas[2]
   - 1.126695627034258 gammas[3] + tau[19,7] = 0
 Ensure_only_one_R_true_19: R[19,0] + R[19,1] + R[19,2] + R[19,3] + R[19,4]
   + R[19,5] + R[19,6] + R[19,7] = 1
 Define_Q_hat_19: - 2503.573704578908 R[19,0] - 2695.835838177543 R[19,1]
   - 2700.478726104334 R[19,2] - 2699.826093262133 R[19,3]
   - 2527.996156202925 R[19,4] - 2527.996188864934 R[19,5]
   - 2507.672598786154 R[19,6] - 2495.880583074765 R[19,7] + Q_hat[19] = 0
 Adjust_Q_hat_19: Q0_var[19] - Q_hat[19] + Q_hat_adjusted[19] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_19: Total_Demand_Up_to_K_minus_1[19]
   - 352.7883000477917 R[19,0] - 610.5504142712473 R[19,1]
   - 937.0718480475368 R[19,2] - 1194.833961676783 R[19,3]
   - 1278.568803441563 R[19,4] - 1536.330917250611 R[19,5]
   - 1746.459998885579 R[19,6] - 1980.356078446152 R[19,7] = 0
 Constr_Total_Demand_from_K_to_T_19: Total_Demand_from_k_to_T[19]
   - 2143.092006016455 R[19,0] - 1885.329891792999 R[19,1]
   - 1558.80845801671 R[19,2] - 1301.046344387464 R[19,3]
   - 1217.311502622684 R[19,4] - 959.5493888136361 R[19,5]
   - 749.4203071786674 R[19,6] - 515.5242276180952 R[19,7] = 0
 Constr_Left_0_diff_aux_19: Sold_0[19] - Q0_var[19] + Left_0_aux_19 = 0
 Constr_Lost_0_diff_aux_19: - Total_Demand_Up_to_K_minus_1[19] + Q0_var[19]
   + Lost_0_aux_19 = 0
 Constr_Q1_plus_left_19: - Left_0[19] + Q1_plus_left[19] - Q1_var[19] = 0
 Constr_Left_1_diff_aux_19: Sold_1[19] - Q1_plus_left[19] + Left_0_aux_19
   = 0
 Constr_Lost_1_diff_aux_19: - Total_Demand_from_k_to_T[19]
   + Q1_plus_left[19] + Lost_0_aux_19 = 0
 Profit_Constraint_19: - 700 Sold_0[19] + 700 Lost_0[19] - 700 Sold_1[19]
   + 300 Left_1[19] + 700 Lost_1[19] + profits_vars[19] = 0
 R420: - 1.618466905718923 alphas[0] - 0.5715876425516639 alphas[1]
   - 0.6122499453658867 alphas[2] + 0.5310019898690563 alphas[3]
   - alphas[4] + f_var[20] = 0
 Q0_upper_bound_20:
   - 2028.416054007841 Fraction_for_second_order_amount[20] + Q0_var[20]
   = 0
 tau_computation_20_0: - betas[0] - 1.618466905718923 gammas[0]
   - 0.5715876425516639 gammas[1] - 0.6122499453658867 gammas[2]
   + 0.5310019898690563 gammas[3] + tau[20,0] = 0
 tau_computation_20_1: - betas[1] - 1.618466905718923 gammas[0]
   - 0.5715876425516639 gammas[1] - 0.6122499453658867 gammas[2]
   + 0.5310019898690563 gammas[3] + tau[20,1] = 0
 tau_computation_20_2: - betas[2] - 1.618466905718923 gammas[0]
   - 0.5715876425516639 gammas[1] - 0.6122499453658867 gammas[2]
   + 0.5310019898690563 gammas[3] + tau[20,2] = 0
 tau_computation_20_3: - betas[3] - 1.618466905718923 gammas[0]
   - 0.5715876425516639 gammas[1] - 0.6122499453658867 gammas[2]
   + 0.5310019898690563 gammas[3] + tau[20,3] = 0
 tau_computation_20_4: - betas[4] - 1.618466905718923 gammas[0]
   - 0.5715876425516639 gammas[1] - 0.6122499453658867 gammas[2]
   + 0.5310019898690563 gammas[3] + tau[20,4] = 0
 tau_computation_20_5: - betas[5] - 1.618466905718923 gammas[0]
   - 0.5715876425516639 gammas[1] - 0.6122499453658867 gammas[2]
   + 0.5310019898690563 gammas[3] + tau[20,5] = 0
 tau_computation_20_6: - betas[6] - 1.618466905718923 gammas[0]
   - 0.5715876425516639 gammas[1] - 0.6122499453658867 gammas[2]
   + 0.5310019898690563 gammas[3] + tau[20,6] = 0
 tau_computation_20_7: - betas[7] - 1.618466905718923 gammas[0]
   - 0.5715876425516639 gammas[1] - 0.6122499453658867 gammas[2]
   + 0.5310019898690563 gammas[3] + tau[20,7] = 0
 Ensure_only_one_R_true_20: R[20,0] + R[20,1] + R[20,2] + R[20,3] + R[20,4]
   + R[20,5] + R[20,6] + R[20,7] = 1
 Define_Q_hat_20: - 2008.759495815442 R[20,0] - 2147.029278268701 R[20,1]
   - 2139.407699563669 R[20,2] - 2138.765592990153 R[20,3]
   - 2118.323751618663 R[20,4] - 2118.323767429469 R[20,5]
   - 1941.023920090492 R[20,6] - 1889.886285788248 R[20,7] + Q_hat[20] = 0
 Adjust_Q_hat_20: Q0_var[20] - Q_hat[20] + Q_hat_adjusted[20] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_20: Total_Demand_Up_to_K_minus_1[20]
   - 193.139052681573 R[20,0] - 400.3845410299301 R[20,1]
   - 609.9481619387043 R[20,2] - 817.1936228445056 R[20,3]
   - 1021.885452998072 R[20,4] - 1229.130913895859 R[20,5]
   - 1317.64182417281 R[20,6] - 1475.395085876261 R[20,7] = 0
 Constr_Total_Demand_from_K_to_T_20: Total_Demand_from_k_to_T[20]
   - 1696.746954990262 R[20,0] - 1489.501466641905 R[20,1]
   - 1279.937845733131 R[20,2] - 1072.69238482733 R[20,3]
   - 868.0005546737632 R[20,4] - 660.7550937759761 R[20,5]
   - 572.2441834990254 R[20,6] - 414.4909217955744 R[20,7] = 0
 Constr_Left_0_diff_aux_20: Sold_0[20] - Q0_var[20] + Left_0_aux_20 = 0
 Constr_Lost_0_diff_aux_20: - Total_Demand_Up_to_K_minus_1[20] + Q0_var[20]
   + Lost_0_aux_20 = 0
 Constr_Q1_plus_left_20: - Left_0[20] + Q1_plus_left[20] - Q1_var[20] = 0
 Constr_Left_1_diff_aux_20: Sold_1[20] - Q1_plus_left[20] + Left_0_aux_20
   = 0
 Constr_Lost_1_diff_aux_20: - Total_Demand_from_k_to_T[20]
   + Q1_plus_left[20] + Lost_0_aux_20 = 0
 Profit_Constraint_20: - 700 Sold_0[20] + 700 Lost_0[20] - 700 Sold_1[20]
   + 300 Left_1[20] + 700 Lost_1[20] + profits_vars[20] = 0
 R441: - 0.959332475660104 alphas[0] + 0.3426726329578249 alphas[1]
   - 1.497506786639079 alphas[2] + 1.034954163865374 alphas[3] - alphas[4]
   + f_var[21] = 0
 Q0_upper_bound_21:
   - 2028.416054007841 Fraction_for_second_order_amount[21] + Q0_var[21]
   = 0
 tau_computation_21_0: - betas[0] - 0.959332475660104 gammas[0]
   + 0.3426726329578249 gammas[1] - 1.497506786639079 gammas[2]
   + 1.034954163865374 gammas[3] + tau[21,0] = 0
 tau_computation_21_1: - betas[1] - 0.959332475660104 gammas[0]
   + 0.3426726329578249 gammas[1] - 1.497506786639079 gammas[2]
   + 1.034954163865374 gammas[3] + tau[21,1] = 0
 tau_computation_21_2: - betas[2] - 0.959332475660104 gammas[0]
   + 0.3426726329578249 gammas[1] - 1.497506786639079 gammas[2]
   + 1.034954163865374 gammas[3] + tau[21,2] = 0
 tau_computation_21_3: - betas[3] - 0.959332475660104 gammas[0]
   + 0.3426726329578249 gammas[1] - 1.497506786639079 gammas[2]
   + 1.034954163865374 gammas[3] + tau[21,3] = 0
 tau_computation_21_4: - betas[4] - 0.959332475660104 gammas[0]
   + 0.3426726329578249 gammas[1] - 1.497506786639079 gammas[2]
   + 1.034954163865374 gammas[3] + tau[21,4] = 0
 tau_computation_21_5: - betas[5] - 0.959332475660104 gammas[0]
   + 0.3426726329578249 gammas[1] - 1.497506786639079 gammas[2]
   + 1.034954163865374 gammas[3] + tau[21,5] = 0
 tau_computation_21_6: - betas[6] - 0.959332475660104 gammas[0]
   + 0.3426726329578249 gammas[1] - 1.497506786639079 gammas[2]
   + 1.034954163865374 gammas[3] + tau[21,6] = 0
 tau_computation_21_7: - betas[7] - 0.959332475660104 gammas[0]
   + 0.3426726329578249 gammas[1] - 1.497506786639079 gammas[2]
   + 1.034954163865374 gammas[3] + tau[21,7] = 0
 Ensure_only_one_R_true_21: R[21,0] + R[21,1] + R[21,2] + R[21,3] + R[21,4]
   + R[21,5] + R[21,6] + R[21,7] = 1
 Define_Q_hat_21: - 1833.805921161819 R[21,0] - 2670.869575784614 R[21,1]
   - 2646.440842933776 R[21,2] - 2654.000592419775 R[21,3]
   - 2590.723281194348 R[21,4] - 2590.72335679779 R[21,5]
   - 2621.130535487543 R[21,6] - 2533.392768903952 R[21,7] + Q_hat[21] = 0
 Adjust_Q_hat_21: Q0_var[21] - Q_hat[21] + Q_hat_adjusted[21] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_21: Total_Demand_Up_to_K_minus_1[21]
   - 136.6911859412865 R[21,0] - 404.8538499338268 R[21,1]
   - 595.6578809278353 R[21,2] - 863.8205448032601 R[21,3]
   - 1152.640679827451 R[21,4] - 1420.8033436893 R[21,5]
   - 1751.029253391662 R[21,6] - 1997.067163955949 R[21,7] = 0
 Constr_Total_Demand_from_K_to_T_21: Total_Demand_from_k_to_T[21]
   - 2396.701305738361 R[21,0] - 2128.53864174582 R[21,1]
   - 1937.734610751812 R[21,2] - 1669.571946876387 R[21,3]
   - 1380.751811852196 R[21,4] - 1112.589147990347 R[21,5]
   - 782.363238287985 R[21,6] - 536.3253277236977 R[21,7] = 0
 Constr_Left_0_diff_aux_21: Sold_0[21] - Q0_var[21] + Left_0_aux_21 = 0
 Constr_Lost_0_diff_aux_21: - Total_Demand_Up_to_K_minus_1[21] + Q0_var[21]
   + Lost_0_aux_21 = 0
 Constr_Q1_plus_left_21: - Left_0[21] + Q1_plus_left[21] - Q1_var[21] = 0
 Constr_Left_1_diff_aux_21: Sold_1[21] - Q1_plus_left[21] + Left_0_aux_21
   = 0
 Constr_Lost_1_diff_aux_21: - Total_Demand_from_k_to_T[21]
   + Q1_plus_left[21] + Lost_0_aux_21 = 0
 Profit_Constraint_21: - 700 Sold_0[21] + 700 Lost_0[21] - 700 Sold_1[21]
   + 300 Left_1[21] + 700 Lost_1[21] + profits_vars[21] = 0
 R462: 0.2809230890672019 alphas[0] - 1.514016999219016 alphas[1]
   + 0.6790746192443368 alphas[2] + 1.31119748641023 alphas[3] - alphas[4]
   + f_var[22] = 0
 Q0_upper_bound_22:
   - 2028.416054007841 Fraction_for_second_order_amount[22] + Q0_var[22]
   = 0
 tau_computation_22_0: - betas[0] + 0.2809230890672019 gammas[0]
   - 1.514016999219016 gammas[1] + 0.6790746192443368 gammas[2]
   + 1.31119748641023 gammas[3] + tau[22,0] = 0
 tau_computation_22_1: - betas[1] + 0.2809230890672019 gammas[0]
   - 1.514016999219016 gammas[1] + 0.6790746192443368 gammas[2]
   + 1.31119748641023 gammas[3] + tau[22,1] = 0
 tau_computation_22_2: - betas[2] + 0.2809230890672019 gammas[0]
   - 1.514016999219016 gammas[1] + 0.6790746192443368 gammas[2]
   + 1.31119748641023 gammas[3] + tau[22,2] = 0
 tau_computation_22_3: - betas[3] + 0.2809230890672019 gammas[0]
   - 1.514016999219016 gammas[1] + 0.6790746192443368 gammas[2]
   + 1.31119748641023 gammas[3] + tau[22,3] = 0
 tau_computation_22_4: - betas[4] + 0.2809230890672019 gammas[0]
   - 1.514016999219016 gammas[1] + 0.6790746192443368 gammas[2]
   + 1.31119748641023 gammas[3] + tau[22,4] = 0
 tau_computation_22_5: - betas[5] + 0.2809230890672019 gammas[0]
   - 1.514016999219016 gammas[1] + 0.6790746192443368 gammas[2]
   + 1.31119748641023 gammas[3] + tau[22,5] = 0
 tau_computation_22_6: - betas[6] + 0.2809230890672019 gammas[0]
   - 1.514016999219016 gammas[1] + 0.6790746192443368 gammas[2]
   + 1.31119748641023 gammas[3] + tau[22,6] = 0
 tau_computation_22_7: - betas[7] + 0.2809230890672019 gammas[0]
   - 1.514016999219016 gammas[1] + 0.6790746192443368 gammas[2]
   + 1.31119748641023 gammas[3] + tau[22,7] = 0
 Ensure_only_one_R_true_22: R[22,0] + R[22,1] + R[22,2] + R[22,3] + R[22,4]
   + R[22,5] + R[22,6] + R[22,7] = 1
 Define_Q_hat_22: - 1951.268278433998 R[22,0] - 1327.235780315859 R[22,1]
   - 1384.040991302919 R[22,2] - 1371.076137304661 R[22,3]
   - 1404.548632811604 R[22,4] - 1404.54853755325 R[22,5]
   - 1439.487154164219 R[22,6] - 1327.848424262717 R[22,7] + Q_hat[22] = 0
 Adjust_Q_hat_22: Q0_var[22] - Q_hat[22] + Q_hat_adjusted[22] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_22: Total_Demand_Up_to_K_minus_1[22]
   - 174.5898084954533 R[22,0] - 293.0014337901478 R[22,1]
   - 570.8379538442234 R[22,2] - 689.2234624818939 R[22,3]
   - 786.7015363450649 R[22,4] - 905.0870450489739 R[22,5]
   - 991.0552742705994 R[22,6] - 1091.077127926938 R[22,7] = 0
 Constr_Total_Demand_from_K_to_T_22: Total_Demand_from_k_to_T[22]
   - 1153.258336839279 R[22,0] - 1034.846711544584 R[22,1]
   - 757.0101914905084 R[22,2] - 638.624682852838 R[22,3]
   - 541.1466089896668 R[22,4] - 422.7611002857579 R[22,5]
   - 336.7928710641324 R[22,6] - 236.7710174077933 R[22,7] = 0
 Constr_Left_0_diff_aux_22: Sold_0[22] - Q0_var[22] + Left_0_aux_22 = 0
 Constr_Lost_0_diff_aux_22: - Total_Demand_Up_to_K_minus_1[22] + Q0_var[22]
   + Lost_0_aux_22 = 0
 Constr_Q1_plus_left_22: - Left_0[22] + Q1_plus_left[22] - Q1_var[22] = 0
 Constr_Left_1_diff_aux_22: Sold_1[22] - Q1_plus_left[22] + Left_0_aux_22
   = 0
 Constr_Lost_1_diff_aux_22: - Total_Demand_from_k_to_T[22]
   + Q1_plus_left[22] + Lost_0_aux_22 = 0
 Profit_Constraint_22: - 700 Sold_0[22] + 700 Lost_0[22] - 700 Sold_1[22]
   + 300 Left_1[22] + 700 Lost_1[22] + profits_vars[22] = 0
 R483: - 0.890908948497018 alphas[0] - 0.2398029125610173 alphas[1]
   - 1.226468639016819 alphas[2] + 0.4175017560904177 alphas[3] - alphas[4]
   + f_var[23] = 0
 Q0_upper_bound_23:
   - 2028.416054007841 Fraction_for_second_order_amount[23] + Q0_var[23]
   = 0
 tau_computation_23_0: - betas[0] - 0.890908948497018 gammas[0]
   - 0.2398029125610173 gammas[1] - 1.226468639016819 gammas[2]
   + 0.4175017560904177 gammas[3] + tau[23,0] = 0
 tau_computation_23_1: - betas[1] - 0.890908948497018 gammas[0]
   - 0.2398029125610173 gammas[1] - 1.226468639016819 gammas[2]
   + 0.4175017560904177 gammas[3] + tau[23,1] = 0
 tau_computation_23_2: - betas[2] - 0.890908948497018 gammas[0]
   - 0.2398029125610173 gammas[1] - 1.226468639016819 gammas[2]
   + 0.4175017560904177 gammas[3] + tau[23,2] = 0
 tau_computation_23_3: - betas[3] - 0.890908948497018 gammas[0]
   - 0.2398029125610173 gammas[1] - 1.226468639016819 gammas[2]
   + 0.4175017560904177 gammas[3] + tau[23,3] = 0
 tau_computation_23_4: - betas[4] - 0.890908948497018 gammas[0]
   - 0.2398029125610173 gammas[1] - 1.226468639016819 gammas[2]
   + 0.4175017560904177 gammas[3] + tau[23,4] = 0
 tau_computation_23_5: - betas[5] - 0.890908948497018 gammas[0]
   - 0.2398029125610173 gammas[1] - 1.226468639016819 gammas[2]
   + 0.4175017560904177 gammas[3] + tau[23,5] = 0
 tau_computation_23_6: - betas[6] - 0.890908948497018 gammas[0]
   - 0.2398029125610173 gammas[1] - 1.226468639016819 gammas[2]
   + 0.4175017560904177 gammas[3] + tau[23,6] = 0
 tau_computation_23_7: - betas[7] - 0.890908948497018 gammas[0]
   - 0.2398029125610173 gammas[1] - 1.226468639016819 gammas[2]
   + 0.4175017560904177 gammas[3] + tau[23,7] = 0
 Ensure_only_one_R_true_23: R[23,0] + R[23,1] + R[23,2] + R[23,3] + R[23,4]
   + R[23,5] + R[23,6] + R[23,7] = 1
 Define_Q_hat_23: - 2488.317729700416 R[23,0] - 2617.924977367592 R[23,1]
   - 2683.33245893755 R[23,2] - 2670.0174168549 R[23,3]
   - 2700.376181616914 R[23,4] - 2700.376191488502 R[23,5]
   - 2721.744654400587 R[23,6] - 2644.621527966629 R[23,7] + Q_hat[23] = 0
 Adjust_Q_hat_23: Q0_var[23] - Q_hat[23] + Q_hat_adjusted[23] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_23: Total_Demand_Up_to_K_minus_1[23]
   - 347.8660385927788 R[23,0] - 597.3777472437051 R[23,1]
   - 1038.790145413306 R[23,2] - 1288.301854499469 R[23,3]
   - 1481.593050531209 R[23,4] - 1731.10475900935 R[23,5]
   - 1909.46839782918 R[23,6] - 2145.597831988234 R[23,7] = 0
 Constr_Total_Demand_from_K_to_T_23: Total_Demand_from_k_to_T[23]
   - 2296.755210351737 R[23,0] - 2047.24350170081 R[23,1]
   - 1605.83110353121 R[23,2] - 1356.319394445046 R[23,3]
   - 1163.028198413306 R[23,4] - 913.5164899351653 R[23,5]
   - 735.1528511153361 R[23,6] - 499.0234169562818 R[23,7] = 0
 Constr_Left_0_diff_aux_23: Sold_0[23] - Q0_var[23] + Left_0_aux_23 = 0
 Constr_Lost_0_diff_aux_23: - Total_Demand_Up_to_K_minus_1[23] + Q0_var[23]
   + Lost_0_aux_23 = 0
 Constr_Q1_plus_left_23: - Left_0[23] + Q1_plus_left[23] - Q1_var[23] = 0
 Constr_Left_1_diff_aux_23: Sold_1[23] - Q1_plus_left[23] + Left_0_aux_23
   = 0
 Constr_Lost_1_diff_aux_23: - Total_Demand_from_k_to_T[23]
   + Q1_plus_left[23] + Lost_0_aux_23 = 0
 Profit_Constraint_23: - 700 Sold_0[23] + 700 Lost_0[23] - 700 Sold_1[23]
   + 300 Left_1[23] + 700 Lost_1[23] + profits_vars[23] = 0
 R504: 1.54147401940782 alphas[0] + 0.6589641810516157 alphas[1]
   + 0.9984403493944379 alphas[2] + 0.5608001189097406 alphas[3]
   - alphas[4] + f_var[24] = 0
 Q0_upper_bound_24:
   - 2028.416054007841 Fraction_for_second_order_amount[24] + Q0_var[24]
   = 0
 tau_computation_24_0: - betas[0] + 1.54147401940782 gammas[0]
   + 0.6589641810516157 gammas[1] + 0.9984403493944379 gammas[2]
   + 0.5608001189097406 gammas[3] + tau[24,0] = 0
 tau_computation_24_1: - betas[1] + 1.54147401940782 gammas[0]
   + 0.6589641810516157 gammas[1] + 0.9984403493944379 gammas[2]
   + 0.5608001189097406 gammas[3] + tau[24,1] = 0
 tau_computation_24_2: - betas[2] + 1.54147401940782 gammas[0]
   + 0.6589641810516157 gammas[1] + 0.9984403493944379 gammas[2]
   + 0.5608001189097406 gammas[3] + tau[24,2] = 0
 tau_computation_24_3: - betas[3] + 1.54147401940782 gammas[0]
   + 0.6589641810516157 gammas[1] + 0.9984403493944379 gammas[2]
   + 0.5608001189097406 gammas[3] + tau[24,3] = 0
 tau_computation_24_4: - betas[4] + 1.54147401940782 gammas[0]
   + 0.6589641810516157 gammas[1] + 0.9984403493944379 gammas[2]
   + 0.5608001189097406 gammas[3] + tau[24,4] = 0
 tau_computation_24_5: - betas[5] + 1.54147401940782 gammas[0]
   + 0.6589641810516157 gammas[1] + 0.9984403493944379 gammas[2]
   + 0.5608001189097406 gammas[3] + tau[24,5] = 0
 tau_computation_24_6: - betas[6] + 1.54147401940782 gammas[0]
   + 0.6589641810516157 gammas[1] + 0.9984403493944379 gammas[2]
   + 0.5608001189097406 gammas[3] + tau[24,6] = 0
 tau_computation_24_7: - betas[7] + 1.54147401940782 gammas[0]
   + 0.6589641810516157 gammas[1] + 0.9984403493944379 gammas[2]
   + 0.5608001189097406 gammas[3] + tau[24,7] = 0
 Ensure_only_one_R_true_24: R[24,0] + R[24,1] + R[24,2] + R[24,3] + R[24,4]
   + R[24,5] + R[24,6] + R[24,7] = 1
 Define_Q_hat_24: - 1799.932561622009 R[24,0] - 1100.07774262383 R[24,1]
   - 1121.110215836834 R[24,2] - 1118.736299552489 R[24,3]
   - 1062.614413737264 R[24,4] - 1062.614325664103 R[24,5]
   - 1031.81389418003 R[24,6] - 996.6573066940407 R[24,7] + Q_hat[24] = 0
 Adjust_Q_hat_24: Q0_var[24] - Q_hat[24] + Q_hat_adjusted[24] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_24: Total_Demand_Up_to_K_minus_1[24]
   - 125.7621215358601 R[24,0] - 222.2151523946935 R[24,1]
   - 396.172689203143 R[24,2] - 492.5816749962584 R[24,3]
   - 534.6884797959071 R[24,4] - 631.0974658723951 R[24,5]
   - 687.3436499325194 R[24,6] - 803.8390560387079 R[24,7] = 0
 Constr_Total_Demand_from_K_to_T_24: Total_Demand_from_k_to_T[24]
   - 870.8949066558237 R[24,0] - 774.4418757969902 R[24,1]
   - 600.4843389885408 R[24,2] - 504.0753531954254 R[24,3]
   - 461.9685483957767 R[24,4] - 365.5595623192886 R[24,5]
   - 309.3133782591643 R[24,6] - 192.8179721529758 R[24,7] = 0
 Constr_Left_0_diff_aux_24: Sold_0[24] - Q0_var[24] + Left_0_aux_24 = 0
 Constr_Lost_0_diff_aux_24: - Total_Demand_Up_to_K_minus_1[24] + Q0_var[24]
   + Lost_0_aux_24 = 0
 Constr_Q1_plus_left_24: - Left_0[24] + Q1_plus_left[24] - Q1_var[24] = 0
 Constr_Left_1_diff_aux_24: Sold_1[24] - Q1_plus_left[24] + Left_0_aux_24
   = 0
 Constr_Lost_1_diff_aux_24: - Total_Demand_from_k_to_T[24]
   + Q1_plus_left[24] + Lost_0_aux_24 = 0
 Profit_Constraint_24: - 700 Sold_0[24] + 700 Lost_0[24] - 700 Sold_1[24]
   + 300 Left_1[24] + 700 Lost_1[24] + profits_vars[24] = 0
 R525: - 0.3744719111560164 alphas[0] - 0.0863020612640085 alphas[1]
   - 1.788661692043826 alphas[2] - 0.1434236163560723 alphas[3] - alphas[4]
   + f_var[25] = 0
 Q0_upper_bound_25:
   - 2028.416054007841 Fraction_for_second_order_amount[25] + Q0_var[25]
   = 0
 tau_computation_25_0: - betas[0] - 0.3744719111560164 gammas[0]
   - 0.0863020612640085 gammas[1] - 1.788661692043826 gammas[2]
   - 0.1434236163560723 gammas[3] + tau[25,0] = 0
 tau_computation_25_1: - betas[1] - 0.3744719111560164 gammas[0]
   - 0.0863020612640085 gammas[1] - 1.788661692043826 gammas[2]
   - 0.1434236163560723 gammas[3] + tau[25,1] = 0
 tau_computation_25_2: - betas[2] - 0.3744719111560164 gammas[0]
   - 0.0863020612640085 gammas[1] - 1.788661692043826 gammas[2]
   - 0.1434236163560723 gammas[3] + tau[25,2] = 0
 tau_computation_25_3: - betas[3] - 0.3744719111560164 gammas[0]
   - 0.0863020612640085 gammas[1] - 1.788661692043826 gammas[2]
   - 0.1434236163560723 gammas[3] + tau[25,3] = 0
 tau_computation_25_4: - betas[4] - 0.3744719111560164 gammas[0]
   - 0.0863020612640085 gammas[1] - 1.788661692043826 gammas[2]
   - 0.1434236163560723 gammas[3] + tau[25,4] = 0
 tau_computation_25_5: - betas[5] - 0.3744719111560164 gammas[0]
   - 0.0863020612640085 gammas[1] - 1.788661692043826 gammas[2]
   - 0.1434236163560723 gammas[3] + tau[25,5] = 0
 tau_computation_25_6: - betas[6] - 0.3744719111560164 gammas[0]
   - 0.0863020612640085 gammas[1] - 1.788661692043826 gammas[2]
   - 0.1434236163560723 gammas[3] + tau[25,6] = 0
 tau_computation_25_7: - betas[7] - 0.3744719111560164 gammas[0]
   - 0.0863020612640085 gammas[1] - 1.788661692043826 gammas[2]
   - 0.1434236163560723 gammas[3] + tau[25,7] = 0
 Ensure_only_one_R_true_25: R[25,0] + R[25,1] + R[25,2] + R[25,3] + R[25,4]
   + R[25,5] + R[25,6] + R[25,7] = 1
 Define_Q_hat_25: - 2145.976911928461 R[25,0] - 2909.255418499749 R[25,1]
   - 2881.624535058041 R[25,2] - 2890.500588215402 R[25,3]
   - 2877.93354599869 R[25,4] - 2877.933570648172 R[25,5]
   - 2806.028781997993 R[25,6] - 2866.277258732909 R[25,7] + Q_hat[25] = 0
 Adjust_Q_hat_25: Q0_var[25] - Q_hat[25] + Q_hat_adjusted[25] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_25: Total_Demand_Up_to_K_minus_1[25]
   - 237.4115428809252 R[25,0] - 525.6094568096954 R[25,1]
   - 761.1266470301202 R[25,2] - 1049.324560919198 R[25,3]
   - 1362.986514252102 R[25,4] - 1651.184428495088 R[25,5]
   - 1926.807543638932 R[25,6] - 2289.88115278488 R[25,7] = 0
 Constr_Total_Demand_from_K_to_T_25: Total_Demand_from_k_to_T[25]
   - 2628.865438389927 R[25,0] - 2340.667524461157 R[25,1]
   - 2105.150334240732 R[25,2] - 1816.952420351654 R[25,3]
   - 1503.29046701875 R[25,4] - 1215.092552775764 R[25,5]
   - 939.4694376319206 R[25,6] - 576.3958284859723 R[25,7] = 0
 Constr_Left_0_diff_aux_25: Sold_0[25] - Q0_var[25] + Left_0_aux_25 = 0
 Constr_Lost_0_diff_aux_25: - Total_Demand_Up_to_K_minus_1[25] + Q0_var[25]
   + Lost_0_aux_25 = 0
 Constr_Q1_plus_left_25: - Left_0[25] + Q1_plus_left[25] - Q1_var[25] = 0
 Constr_Left_1_diff_aux_25: Sold_1[25] - Q1_plus_left[25] + Left_0_aux_25
   = 0
 Constr_Lost_1_diff_aux_25: - Total_Demand_from_k_to_T[25]
   + Q1_plus_left[25] + Lost_0_aux_25 = 0
 Profit_Constraint_25: - 700 Sold_0[25] + 700 Lost_0[25] - 700 Sold_1[25]
   + 300 Left_1[25] + 700 Lost_1[25] + profits_vars[25] = 0
 R546: 1.449362342404136 alphas[0] + 2.311116297307088 alphas[1]
   - 0.824805748044052 alphas[2] - 0.9678537137090004 alphas[3] - alphas[4]
   + f_var[26] = 0
 Q0_upper_bound_26:
   - 2028.416054007841 Fraction_for_second_order_amount[26] + Q0_var[26]
   = 0
 tau_computation_26_0: - betas[0] + 1.449362342404136 gammas[0]
   + 2.311116297307088 gammas[1] - 0.824805748044052 gammas[2]
   - 0.9678537137090004 gammas[3] + tau[26,0] = 0
 tau_computation_26_1: - betas[1] + 1.449362342404136 gammas[0]
   + 2.311116297307088 gammas[1] - 0.824805748044052 gammas[2]
   - 0.9678537137090004 gammas[3] + tau[26,1] = 0
 tau_computation_26_2: - betas[2] + 1.449362342404136 gammas[0]
   + 2.311116297307088 gammas[1] - 0.824805748044052 gammas[2]
   - 0.9678537137090004 gammas[3] + tau[26,2] = 0
 tau_computation_26_3: - betas[3] + 1.449362342404136 gammas[0]
   + 2.311116297307088 gammas[1] - 0.824805748044052 gammas[2]
   - 0.9678537137090004 gammas[3] + tau[26,3] = 0
 tau_computation_26_4: - betas[4] + 1.449362342404136 gammas[0]
   + 2.311116297307088 gammas[1] - 0.824805748044052 gammas[2]
   - 0.9678537137090004 gammas[3] + tau[26,4] = 0
 tau_computation_26_5: - betas[5] + 1.449362342404136 gammas[0]
   + 2.311116297307088 gammas[1] - 0.824805748044052 gammas[2]
   - 0.9678537137090004 gammas[3] + tau[26,5] = 0
 tau_computation_26_6: - betas[6] + 1.449362342404136 gammas[0]
   + 2.311116297307088 gammas[1] - 0.824805748044052 gammas[2]
   - 0.9678537137090004 gammas[3] + tau[26,6] = 0
 tau_computation_26_7: - betas[7] + 1.449362342404136 gammas[0]
   + 2.311116297307088 gammas[1] - 0.824805748044052 gammas[2]
   - 0.9678537137090004 gammas[3] + tau[26,7] = 0
 Ensure_only_one_R_true_26: R[26,0] + R[26,1] + R[26,2] + R[26,3] + R[26,4]
   + R[26,5] + R[26,6] + R[26,7] = 1
 Define_Q_hat_26: - 2273.808171708853 R[26,0] - 2327.70294862958 R[26,1]
   - 2402.527772884753 R[26,2] - 2385.863278132075 R[26,3]
   - 2386.720869846146 R[26,4] - 2386.720822899134 R[26,5]
   - 2479.257589097558 R[26,6] - 2380.225190676716 R[26,7] + Q_hat[26] = 0
 Adjust_Q_hat_26: Q0_var[26] - Q_hat[26] + Q_hat_adjusted[26] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_26: Total_Demand_Up_to_K_minus_1[26]
   - 278.6556382451527 R[26,0] - 500.5277077818641 R[26,1]
   - 917.2995107514321 R[26,2] - 1139.17157971979 R[26,3]
   - 1296.407438707382 R[26,4] - 1518.279507804335 R[26,5]
   - 1711.775018983877 R[26,6] - 1936.480774365079 R[26,7] = 0
 Constr_Total_Demand_from_K_to_T_26: Total_Demand_from_k_to_T[26]
   - 2101.569274313834 R[26,0] - 1879.697204777123 R[26,1]
   - 1462.925401807555 R[26,2] - 1241.053332839197 R[26,3]
   - 1083.817473851605 R[26,4] - 861.9454047546512 R[26,5]
   - 668.4498935751097 R[26,6] - 443.7441381939076 R[26,7] = 0
 Constr_Left_0_diff_aux_26: Sold_0[26] - Q0_var[26] + Left_0_aux_26 = 0
 Constr_Lost_0_diff_aux_26: - Total_Demand_Up_to_K_minus_1[26] + Q0_var[26]
   + Lost_0_aux_26 = 0
 Constr_Q1_plus_left_26: - Left_0[26] + Q1_plus_left[26] - Q1_var[26] = 0
 Constr_Left_1_diff_aux_26: Sold_1[26] - Q1_plus_left[26] + Left_0_aux_26
   = 0
 Constr_Lost_1_diff_aux_26: - Total_Demand_from_k_to_T[26]
   + Q1_plus_left[26] + Lost_0_aux_26 = 0
 Profit_Constraint_26: - 700 Sold_0[26] + 700 Lost_0[26] - 700 Sold_1[26]
   + 300 Left_1[26] + 700 Lost_1[26] + profits_vars[26] = 0
 R567: - 1.493774703007406 alphas[0] + 1.158209523202141 alphas[1]
   + 0.8899153575123454 alphas[2] - 1.012657893691021 alphas[3] - alphas[4]
   + f_var[27] = 0
 Q0_upper_bound_27:
   - 2028.416054007841 Fraction_for_second_order_amount[27] + Q0_var[27]
   = 0
 tau_computation_27_0: - betas[0] - 1.493774703007406 gammas[0]
   + 1.158209523202141 gammas[1] + 0.8899153575123454 gammas[2]
   - 1.012657893691021 gammas[3] + tau[27,0] = 0
 tau_computation_27_1: - betas[1] - 1.493774703007406 gammas[0]
   + 1.158209523202141 gammas[1] + 0.8899153575123454 gammas[2]
   - 1.012657893691021 gammas[3] + tau[27,1] = 0
 tau_computation_27_2: - betas[2] - 1.493774703007406 gammas[0]
   + 1.158209523202141 gammas[1] + 0.8899153575123454 gammas[2]
   - 1.012657893691021 gammas[3] + tau[27,2] = 0
 tau_computation_27_3: - betas[3] - 1.493774703007406 gammas[0]
   + 1.158209523202141 gammas[1] + 0.8899153575123454 gammas[2]
   - 1.012657893691021 gammas[3] + tau[27,3] = 0
 tau_computation_27_4: - betas[4] - 1.493774703007406 gammas[0]
   + 1.158209523202141 gammas[1] + 0.8899153575123454 gammas[2]
   - 1.012657893691021 gammas[3] + tau[27,4] = 0
 tau_computation_27_5: - betas[5] - 1.493774703007406 gammas[0]
   + 1.158209523202141 gammas[1] + 0.8899153575123454 gammas[2]
   - 1.012657893691021 gammas[3] + tau[27,5] = 0
 tau_computation_27_6: - betas[6] - 1.493774703007406 gammas[0]
   + 1.158209523202141 gammas[1] + 0.8899153575123454 gammas[2]
   - 1.012657893691021 gammas[3] + tau[27,6] = 0
 tau_computation_27_7: - betas[7] - 1.493774703007406 gammas[0]
   + 1.158209523202141 gammas[1] + 0.8899153575123454 gammas[2]
   - 1.012657893691021 gammas[3] + tau[27,7] = 0
 Ensure_only_one_R_true_27: R[27,0] + R[27,1] + R[27,2] + R[27,3] + R[27,4]
   + R[27,5] + R[27,6] + R[27,7] = 1
 Define_Q_hat_27: - 1410.147544236772 R[27,0] - 1098.113875584861 R[27,1]
   - 1132.260169467231 R[27,2] - 1119.325865004735 R[27,3]
   - 1175.544073868488 R[27,4] - 1175.544012059556 R[27,5]
   - 1058.851555526858 R[27,6] - 1096.406143860195 R[27,7] + Q_hat[27] = 0
 Adjust_Q_hat_27: Q0_var[27] - Q_hat[27] + Q_hat_adjusted[27] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_27: Total_Demand_Up_to_K_minus_1[27]
   - 103.8847732122259 R[27,1] - 259.882021496053 R[27,2]
   - 363.758940586994 R[27,3] - 544.509136870977 R[27,4]
   - 648.3860561493666 R[27,5] - 648.3860561493666 R[27,6]
   - 888.6520263497487 R[27,7] = 0
 Constr_Total_Demand_from_K_to_T_27: Total_Demand_from_k_to_T[27]
   - 1096.405864906528 R[27,0] - 992.5210916943022 R[27,1]
   - 836.5238434104751 R[27,2] - 732.646924319534 R[27,3]
   - 551.896728035551 R[27,4] - 448.0198087571613 R[27,5]
   - 448.0198087571613 R[27,6] - 207.7538385567793 R[27,7] = 0
 Constr_Left_0_diff_aux_27: Sold_0[27] - Q0_var[27] + Left_0_aux_27 = 0
 Constr_Lost_0_diff_aux_27: - Total_Demand_Up_to_K_minus_1[27] + Q0_var[27]
   + Lost_0_aux_27 = 0
 Constr_Q1_plus_left_27: - Left_0[27] + Q1_plus_left[27] - Q1_var[27] = 0
 Constr_Left_1_diff_aux_27: Sold_1[27] - Q1_plus_left[27] + Left_0_aux_27
   = 0
 Constr_Lost_1_diff_aux_27: - Total_Demand_from_k_to_T[27]
   + Q1_plus_left[27] + Lost_0_aux_27 = 0
 Profit_Constraint_27: - 700 Sold_0[27] + 700 Lost_0[27] - 700 Sold_1[27]
   + 300 Left_1[27] + 700 Lost_1[27] + profits_vars[27] = 0
 R588: 0.0591950776359468 alphas[0] - 0.3058761945937323 alphas[1]
   - 1.768966778466122 alphas[2] + 0.4831241513113188 alphas[3] - alphas[4]
   + f_var[28] = 0
 Q0_upper_bound_28:
   - 2028.416054007841 Fraction_for_second_order_amount[28] + Q0_var[28]
   = 0
 tau_computation_28_0: - betas[0] + 0.0591950776359468 gammas[0]
   - 0.3058761945937323 gammas[1] - 1.768966778466122 gammas[2]
   + 0.4831241513113188 gammas[3] + tau[28,0] = 0
 tau_computation_28_1: - betas[1] + 0.0591950776359468 gammas[0]
   - 0.3058761945937323 gammas[1] - 1.768966778466122 gammas[2]
   + 0.4831241513113188 gammas[3] + tau[28,1] = 0
 tau_computation_28_2: - betas[2] + 0.0591950776359468 gammas[0]
   - 0.3058761945937323 gammas[1] - 1.768966778466122 gammas[2]
   + 0.4831241513113188 gammas[3] + tau[28,2] = 0
 tau_computation_28_3: - betas[3] + 0.0591950776359468 gammas[0]
   - 0.3058761945937323 gammas[1] - 1.768966778466122 gammas[2]
   + 0.4831241513113188 gammas[3] + tau[28,3] = 0
 tau_computation_28_4: - betas[4] + 0.0591950776359468 gammas[0]
   - 0.3058761945937323 gammas[1] - 1.768966778466122 gammas[2]
   + 0.4831241513113188 gammas[3] + tau[28,4] = 0
 tau_computation_28_5: - betas[5] + 0.0591950776359468 gammas[0]
   - 0.3058761945937323 gammas[1] - 1.768966778466122 gammas[2]
   + 0.4831241513113188 gammas[3] + tau[28,5] = 0
 tau_computation_28_6: - betas[6] + 0.0591950776359468 gammas[0]
   - 0.3058761945937323 gammas[1] - 1.768966778466122 gammas[2]
   + 0.4831241513113188 gammas[3] + tau[28,6] = 0
 tau_computation_28_7: - betas[7] + 0.0591950776359468 gammas[0]
   - 0.3058761945937323 gammas[1] - 1.768966778466122 gammas[2]
   + 0.4831241513113188 gammas[3] + tau[28,7] = 0
 Ensure_only_one_R_true_28: R[28,0] + R[28,1] + R[28,2] + R[28,3] + R[28,4]
   + R[28,5] + R[28,6] + R[28,7] = 1
 Define_Q_hat_28: - 2638.726174467007 R[28,0] - 2985.000568942854 R[28,1]
   - 2949.827548481035 R[28,2] - 2958.880309679967 R[28,3]
   - 3099.965873525296 R[28,4] - 3099.965823276414 R[28,5]
   - 3073.318651809615 R[28,6] - 2993.907028851979 R[28,7] + Q_hat[28] = 0
 Adjust_Q_hat_28: Q0_var[28] - Q_hat[28] + Q_hat_adjusted[28] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_28: Total_Demand_Up_to_K_minus_1[28]
   - 396.3945460188629 R[28,0] - 683.2371932766055 R[28,1]
   - 964.4245928373681 R[28,2] - 1251.267240030188 R[28,3]
   - 1616.497481099766 R[28,4] - 1903.340128721997 R[28,5]
   - 2232.046159898827 R[28,6] - 2420.221455590129 R[28,7] = 0
 Constr_Total_Demand_from_K_to_T_28: Total_Demand_from_k_to_T[28]
   - 2597.512204815728 R[28,0] - 2310.669557557986 R[28,1]
   - 2029.482157997223 R[28,2] - 1742.639510804403 R[28,3]
   - 1377.409269734826 R[28,4] - 1090.566622112594 R[28,5]
   - 761.8605909357638 R[28,6] - 573.6852952444622 R[28,7] = 0
 Constr_Left_0_diff_aux_28: Sold_0[28] - Q0_var[28] + Left_0_aux_28 = 0
 Constr_Lost_0_diff_aux_28: - Total_Demand_Up_to_K_minus_1[28] + Q0_var[28]
   + Lost_0_aux_28 = 0
 Constr_Q1_plus_left_28: - Left_0[28] + Q1_plus_left[28] - Q1_var[28] = 0
 Constr_Left_1_diff_aux_28: Sold_1[28] - Q1_plus_left[28] + Left_0_aux_28
   = 0
 Constr_Lost_1_diff_aux_28: - Total_Demand_from_k_to_T[28]
   + Q1_plus_left[28] + Lost_0_aux_28 = 0
 Profit_Constraint_28: - 700 Sold_0[28] + 700 Lost_0[28] - 700 Sold_1[28]
   + 300 Left_1[28] + 700 Lost_1[28] + profits_vars[28] = 0
 R609: 0.4528779152734247 alphas[0] + 1.205150293466185 alphas[1]
   - 0.9823614182491538 alphas[2] + 0.1332012933893546 alphas[3]
   - alphas[4] + f_var[29] = 0
 Q0_upper_bound_29:
   - 2028.416054007841 Fraction_for_second_order_amount[29] + Q0_var[29]
   = 0
 tau_computation_29_0: - betas[0] + 0.4528779152734247 gammas[0]
   + 1.205150293466185 gammas[1] - 0.9823614182491538 gammas[2]
   + 0.1332012933893546 gammas[3] + tau[29,0] = 0
 tau_computation_29_1: - betas[1] + 0.4528779152734247 gammas[0]
   + 1.205150293466185 gammas[1] - 0.9823614182491538 gammas[2]
   + 0.1332012933893546 gammas[3] + tau[29,1] = 0
 tau_computation_29_2: - betas[2] + 0.4528779152734247 gammas[0]
   + 1.205150293466185 gammas[1] - 0.9823614182491538 gammas[2]
   + 0.1332012933893546 gammas[3] + tau[29,2] = 0
 tau_computation_29_3: - betas[3] + 0.4528779152734247 gammas[0]
   + 1.205150293466185 gammas[1] - 0.9823614182491538 gammas[2]
   + 0.1332012933893546 gammas[3] + tau[29,3] = 0
 tau_computation_29_4: - betas[4] + 0.4528779152734247 gammas[0]
   + 1.205150293466185 gammas[1] - 0.9823614182491538 gammas[2]
   + 0.1332012933893546 gammas[3] + tau[29,4] = 0
 tau_computation_29_5: - betas[5] + 0.4528779152734247 gammas[0]
   + 1.205150293466185 gammas[1] - 0.9823614182491538 gammas[2]
   + 0.1332012933893546 gammas[3] + tau[29,5] = 0
 tau_computation_29_6: - betas[6] + 0.4528779152734247 gammas[0]
   + 1.205150293466185 gammas[1] - 0.9823614182491538 gammas[2]
   + 0.1332012933893546 gammas[3] + tau[29,6] = 0
 tau_computation_29_7: - betas[7] + 0.4528779152734247 gammas[0]
   + 1.205150293466185 gammas[1] - 0.9823614182491538 gammas[2]
   + 0.1332012933893546 gammas[3] + tau[29,7] = 0
 Ensure_only_one_R_true_29: R[29,0] + R[29,1] + R[29,2] + R[29,3] + R[29,4]
   + R[29,5] + R[29,6] + R[29,7] = 1
 Define_Q_hat_29: - 1764.630615603476 R[29,0] - 2335.469232972943 R[29,1]
   - 2354.271449585769 R[29,2] - 2350.818551491291 R[29,3]
   - 2110.007700326413 R[29,4] - 2110.007759786818 R[29,5]
   - 1965.455745636935 R[29,6] - 1881.124491514184 R[29,7] + Q_hat[29] = 0
 Adjust_Q_hat_29: Q0_var[29] - Q_hat[29] + Q_hat_adjusted[29] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_29: Total_Demand_Up_to_K_minus_1[29]
   - 114.3721310856955 R[29,0] - 347.0860823021803 R[29,1]
   - 594.470270702276 R[29,2] - 827.1842220057102 R[29,3]
   - 917.6151705794235 R[29,4] - 1150.329122271963 R[29,5]
   - 1228.715845487517 R[29,6] - 1415.696311696586 R[29,7] = 0
 Constr_Total_Demand_from_K_to_T_29: Total_Demand_from_k_to_T[29]
   - 1766.75208399597 R[29,0] - 1534.038132779485 R[29,1]
   - 1286.653944379389 R[29,2] - 1053.939993075955 R[29,3]
   - 963.5090445022415 R[29,4] - 730.7950928097021 R[29,5]
   - 652.4083695941483 R[29,6] - 465.4279033850789 R[29,7] = 0
 Constr_Left_0_diff_aux_29: Sold_0[29] - Q0_var[29] + Left_0_aux_29 = 0
 Constr_Lost_0_diff_aux_29: - Total_Demand_Up_to_K_minus_1[29] + Q0_var[29]
   + Lost_0_aux_29 = 0
 Constr_Q1_plus_left_29: - Left_0[29] + Q1_plus_left[29] - Q1_var[29] = 0
 Constr_Left_1_diff_aux_29: Sold_1[29] - Q1_plus_left[29] + Left_0_aux_29
   = 0
 Constr_Lost_1_diff_aux_29: - Total_Demand_from_k_to_T[29]
   + Q1_plus_left[29] + Lost_0_aux_29 = 0
 Profit_Constraint_29: - 700 Sold_0[29] + 700 Lost_0[29] - 700 Sold_1[29]
   + 300 Left_1[29] + 700 Lost_1[29] + profits_vars[29] = 0
 R630: 1.004200612610371 alphas[0] - 0.1337532032057398 alphas[1]
   + 0.750283491536 alphas[2] + 0.854678609519227 alphas[3] - alphas[4]
   + f_var[30] = 0
 Q0_upper_bound_30:
   - 2028.416054007841 Fraction_for_second_order_amount[30] + Q0_var[30]
   = 0
 tau_computation_30_0: - betas[0] + 1.004200612610371 gammas[0]
   - 0.1337532032057398 gammas[1] + 0.750283491536 gammas[2]
   + 0.854678609519227 gammas[3] + tau[30,0] = 0
 tau_computation_30_1: - betas[1] + 1.004200612610371 gammas[0]
   - 0.1337532032057398 gammas[1] + 0.750283491536 gammas[2]
   + 0.854678609519227 gammas[3] + tau[30,1] = 0
 tau_computation_30_2: - betas[2] + 1.004200612610371 gammas[0]
   - 0.1337532032057398 gammas[1] + 0.750283491536 gammas[2]
   + 0.854678609519227 gammas[3] + tau[30,2] = 0
 tau_computation_30_3: - betas[3] + 1.004200612610371 gammas[0]
   - 0.1337532032057398 gammas[1] + 0.750283491536 gammas[2]
   + 0.854678609519227 gammas[3] + tau[30,3] = 0
 tau_computation_30_4: - betas[4] + 1.004200612610371 gammas[0]
   - 0.1337532032057398 gammas[1] + 0.750283491536 gammas[2]
   + 0.854678609519227 gammas[3] + tau[30,4] = 0
 tau_computation_30_5: - betas[5] + 1.004200612610371 gammas[0]
   - 0.1337532032057398 gammas[1] + 0.750283491536 gammas[2]
   + 0.854678609519227 gammas[3] + tau[30,5] = 0
 tau_computation_30_6: - betas[6] + 1.004200612610371 gammas[0]
   - 0.1337532032057398 gammas[1] + 0.750283491536 gammas[2]
   + 0.854678609519227 gammas[3] + tau[30,6] = 0
 tau_computation_30_7: - betas[7] + 1.004200612610371 gammas[0]
   - 0.1337532032057398 gammas[1] + 0.750283491536 gammas[2]
   + 0.854678609519227 gammas[3] + tau[30,7] = 0
 Ensure_only_one_R_true_30: R[30,0] + R[30,1] + R[30,2] + R[30,3] + R[30,4]
   + R[30,5] + R[30,6] + R[30,7] = 1
 Define_Q_hat_30: - 2233.015812934778 R[30,0] - 1332.819772541182 R[30,1]
   - 1302.504620656697 R[30,2] - 1301.967456133353 R[30,3]
   - 1202.504166248167 R[30,4] - 1202.504190730054 R[30,5]
   - 1177.656358520783 R[30,6] - 1096.732866974807 R[30,7] + Q_hat[30] = 0
 Adjust_Q_hat_30: Q0_var[30] - Q_hat[30] + Q_hat_adjusted[30] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_30: Total_Demand_Up_to_K_minus_1[30]
   - 265.4941944926616 R[30,0] - 378.9909434078693 R[30,1]
   - 514.4799725311365 R[30,2] - 627.9653837335989 R[30,3]
   - 627.9653837335989 R[30,4] - 741.4507943823553 R[30,5]
   - 869.7617671527879 R[30,6] - 869.7617671527879 R[30,7] = 0
 Constr_Total_Demand_from_K_to_T_30: Total_Demand_from_k_to_T[30]
   - 831.2383939576387 R[30,0] - 717.7416450424309 R[30,1]
   - 582.2526159191639 R[30,2] - 468.7672047167014 R[30,3]
   - 468.7672047167014 R[30,4] - 355.281794067945 R[30,5]
   - 226.9708212975124 R[30,6] - 226.9708212975124 R[30,7] = 0
 Constr_Left_0_diff_aux_30: Sold_0[30] - Q0_var[30] + Left_0_aux_30 = 0
 Constr_Lost_0_diff_aux_30: - Total_Demand_Up_to_K_minus_1[30] + Q0_var[30]
   + Lost_0_aux_30 = 0
 Constr_Q1_plus_left_30: - Left_0[30] + Q1_plus_left[30] - Q1_var[30] = 0
 Constr_Left_1_diff_aux_30: Sold_1[30] - Q1_plus_left[30] + Left_0_aux_30
   = 0
 Constr_Lost_1_diff_aux_30: - Total_Demand_from_k_to_T[30]
   + Q1_plus_left[30] + Lost_0_aux_30 = 0
 Profit_Constraint_30: - 700 Sold_0[30] + 700 Lost_0[30] - 700 Sold_1[30]
   + 300 Left_1[30] + 700 Lost_1[30] + profits_vars[30] = 0
 R651: - 0.8677863737108036 alphas[0] + 0.6389027366982564 alphas[1]
   + 0.8978924072085647 alphas[2] - 1.349920552576007 alphas[3] - alphas[4]
   + f_var[31] = 0
 Q0_upper_bound_31:
   - 2028.416054007841 Fraction_for_second_order_amount[31] + Q0_var[31]
   = 0
 tau_computation_31_0: - betas[0] - 0.8677863737108036 gammas[0]
   + 0.6389027366982564 gammas[1] + 0.8978924072085647 gammas[2]
   - 1.349920552576007 gammas[3] + tau[31,0] = 0
 tau_computation_31_1: - betas[1] - 0.8677863737108036 gammas[0]
   + 0.6389027366982564 gammas[1] + 0.8978924072085647 gammas[2]
   - 1.349920552576007 gammas[3] + tau[31,1] = 0
 tau_computation_31_2: - betas[2] - 0.8677863737108036 gammas[0]
   + 0.6389027366982564 gammas[1] + 0.8978924072085647 gammas[2]
   - 1.349920552576007 gammas[3] + tau[31,2] = 0
 tau_computation_31_3: - betas[3] - 0.8677863737108036 gammas[0]
   + 0.6389027366982564 gammas[1] + 0.8978924072085647 gammas[2]
   - 1.349920552576007 gammas[3] + tau[31,3] = 0
 tau_computation_31_4: - betas[4] - 0.8677863737108036 gammas[0]
   + 0.6389027366982564 gammas[1] + 0.8978924072085647 gammas[2]
   - 1.349920552576007 gammas[3] + tau[31,4] = 0
 tau_computation_31_5: - betas[5] - 0.8677863737108036 gammas[0]
   + 0.6389027366982564 gammas[1] + 0.8978924072085647 gammas[2]
   - 1.349920552576007 gammas[3] + tau[31,5] = 0
 tau_computation_31_6: - betas[6] - 0.8677863737108036 gammas[0]
   + 0.6389027366982564 gammas[1] + 0.8978924072085647 gammas[2]
   - 1.349920552576007 gammas[3] + tau[31,6] = 0
 tau_computation_31_7: - betas[7] - 0.8677863737108036 gammas[0]
   + 0.6389027366982564 gammas[1] + 0.8978924072085647 gammas[2]
   - 1.349920552576007 gammas[3] + tau[31,7] = 0
 Ensure_only_one_R_true_31: R[31,0] + R[31,1] + R[31,2] + R[31,3] + R[31,4]
   + R[31,5] + R[31,6] + R[31,7] = 1
 Define_Q_hat_31: - 1980.076387489474 R[31,0] - 1194.951727162148 R[31,1]
   - 1167.766438773825 R[31,2] - 1164.04768934455 R[31,3]
   - 1102.594728972929 R[31,4] - 1102.594713342965 R[31,5]
   - 1220.093311822981 R[31,6] - 1160.339737466994 R[31,7] + Q_hat[31] = 0
 Adjust_Q_hat_31: Q0_var[31] - Q_hat[31] + Q_hat_adjusted[31] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_31: Total_Demand_Up_to_K_minus_1[31]
   - 183.884596008213 R[31,0] - 287.2145572737605 R[31,1]
   - 391.2119626810434 R[31,2] - 494.5399570156345 R[31,3]
   - 541.727474194637 R[31,4] - 645.0554685365075 R[31,5]
   - 878.2738479268712 R[31,6] - 953.6834708503509 R[31,7] = 0
 Constr_Total_Demand_from_K_to_T_31: Total_Demand_from_k_to_T[31]
   - 976.4548635258785 R[31,0] - 873.1249022603309 R[31,1]
   - 769.1274968530481 R[31,2] - 665.799502518457 R[31,3]
   - 618.6119853394543 R[31,4] - 515.2839909975839 R[31,5]
   - 282.0656116072201 R[31,6] - 206.6559886837406 R[31,7] = 0
 Constr_Left_0_diff_aux_31: Sold_0[31] - Q0_var[31] + Left_0_aux_31 = 0
 Constr_Lost_0_diff_aux_31: - Total_Demand_Up_to_K_minus_1[31] + Q0_var[31]
   + Lost_0_aux_31 = 0
 Constr_Q1_plus_left_31: - Left_0[31] + Q1_plus_left[31] - Q1_var[31] = 0
 Constr_Left_1_diff_aux_31: Sold_1[31] - Q1_plus_left[31] + Left_0_aux_31
   = 0
 Constr_Lost_1_diff_aux_31: - Total_Demand_from_k_to_T[31]
   + Q1_plus_left[31] + Lost_0_aux_31 = 0
 Profit_Constraint_31: - 700 Sold_0[31] + 700 Lost_0[31] - 700 Sold_1[31]
   + 300 Left_1[31] + 700 Lost_1[31] + profits_vars[31] = 0
 R672: 0.3004959828634036 alphas[0] + 1.8363651474761 alphas[1]
   - 0.2097784424729602 alphas[2] + 1.305848341828286 alphas[3] - alphas[4]
   + f_var[32] = 0
 Q0_upper_bound_32:
   - 2028.416054007841 Fraction_for_second_order_amount[32] + Q0_var[32]
   = 0
 tau_computation_32_0: - betas[0] + 0.3004959828634036 gammas[0]
   + 1.8363651474761 gammas[1] - 0.2097784424729602 gammas[2]
   + 1.305848341828286 gammas[3] + tau[32,0] = 0
 tau_computation_32_1: - betas[1] + 0.3004959828634036 gammas[0]
   + 1.8363651474761 gammas[1] - 0.2097784424729602 gammas[2]
   + 1.305848341828286 gammas[3] + tau[32,1] = 0
 tau_computation_32_2: - betas[2] + 0.3004959828634036 gammas[0]
   + 1.8363651474761 gammas[1] - 0.2097784424729602 gammas[2]
   + 1.305848341828286 gammas[3] + tau[32,2] = 0
 tau_computation_32_3: - betas[3] + 0.3004959828634036 gammas[0]
   + 1.8363651474761 gammas[1] - 0.2097784424729602 gammas[2]
   + 1.305848341828286 gammas[3] + tau[32,3] = 0
 tau_computation_32_4: - betas[4] + 0.3004959828634036 gammas[0]
   + 1.8363651474761 gammas[1] - 0.2097784424729602 gammas[2]
   + 1.305848341828286 gammas[3] + tau[32,4] = 0
 tau_computation_32_5: - betas[5] + 0.3004959828634036 gammas[0]
   + 1.8363651474761 gammas[1] - 0.2097784424729602 gammas[2]
   + 1.305848341828286 gammas[3] + tau[32,5] = 0
 tau_computation_32_6: - betas[6] + 0.3004959828634036 gammas[0]
   + 1.8363651474761 gammas[1] - 0.2097784424729602 gammas[2]
   + 1.305848341828286 gammas[3] + tau[32,6] = 0
 tau_computation_32_7: - betas[7] + 0.3004959828634036 gammas[0]
   + 1.8363651474761 gammas[1] - 0.2097784424729602 gammas[2]
   + 1.305848341828286 gammas[3] + tau[32,7] = 0
 Ensure_only_one_R_true_32: R[32,0] + R[32,1] + R[32,2] + R[32,3] + R[32,4]
   + R[32,5] + R[32,6] + R[32,7] = 1
 Define_Q_hat_32: - 1873.107593697431 R[32,0] - 1870.398710746891 R[32,1]
   - 1920.952747704457 R[32,2] - 1907.053743674092 R[32,3]
   - 1799.316178690225 R[32,4] - 1799.316189078475 R[32,5]
   - 1817.741231394396 R[32,6] - 1769.20181781339 R[32,7] + Q_hat[32] = 0
 Adjust_Q_hat_32: Q0_var[32] - Q_hat[32] + Q_hat_adjusted[32] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_32: Total_Demand_Up_to_K_minus_1[32]
   - 149.3716674824772 R[32,0] - 328.9218844314183 R[32,1]
   - 621.389698307788 R[32,2] - 800.9398768263 R[32,3]
   - 886.8817875972566 R[32,4] - 1066.431966079922 R[32,5]
   - 1188.797433166875 R[32,6] - 1410.101183009175 R[32,7] = 0
 Constr_Total_Demand_from_K_to_T_32: Total_Demand_from_k_to_T[32]
   - 1619.829872492029 R[32,0] - 1440.279655543088 R[32,1]
   - 1147.811841666719 R[32,2] - 968.2616631482066 R[32,3]
   - 882.31975237725 R[32,4] - 702.7695738945841 R[32,5]
   - 580.4041068076316 R[32,6] - 359.1003569653316 R[32,7] = 0
 Constr_Left_0_diff_aux_32: Sold_0[32] - Q0_var[32] + Left_0_aux_32 = 0
 Constr_Lost_0_diff_aux_32: - Total_Demand_Up_to_K_minus_1[32] + Q0_var[32]
   + Lost_0_aux_32 = 0
 Constr_Q1_plus_left_32: - Left_0[32] + Q1_plus_left[32] - Q1_var[32] = 0
 Constr_Left_1_diff_aux_32: Sold_1[32] - Q1_plus_left[32] + Left_0_aux_32
   = 0
 Constr_Lost_1_diff_aux_32: - Total_Demand_from_k_to_T[32]
   + Q1_plus_left[32] + Lost_0_aux_32 = 0
 Profit_Constraint_32: - 700 Sold_0[32] + 700 Lost_0[32] - 700 Sold_1[32]
   + 300 Left_1[32] + 700 Lost_1[32] + profits_vars[32] = 0
 R693: - 0.1119083925633281 alphas[0] + 1.171652093844782 alphas[1]
   + 1.579628363899807 alphas[2] - 1.017636964542434 alphas[3] - alphas[4]
   + f_var[33] = 0
 Q0_upper_bound_33:
   - 2028.416054007841 Fraction_for_second_order_amount[33] + Q0_var[33]
   = 0
 tau_computation_33_0: - betas[0] - 0.1119083925633281 gammas[0]
   + 1.171652093844782 gammas[1] + 1.579628363899807 gammas[2]
   - 1.017636964542434 gammas[3] + tau[33,0] = 0
 tau_computation_33_1: - betas[1] - 0.1119083925633281 gammas[0]
   + 1.171652093844782 gammas[1] + 1.579628363899807 gammas[2]
   - 1.017636964542434 gammas[3] + tau[33,1] = 0
 tau_computation_33_2: - betas[2] - 0.1119083925633281 gammas[0]
   + 1.171652093844782 gammas[1] + 1.579628363899807 gammas[2]
   - 1.017636964542434 gammas[3] + tau[33,2] = 0
 tau_computation_33_3: - betas[3] - 0.1119083925633281 gammas[0]
   + 1.171652093844782 gammas[1] + 1.579628363899807 gammas[2]
   - 1.017636964542434 gammas[3] + tau[33,3] = 0
 tau_computation_33_4: - betas[4] - 0.1119083925633281 gammas[0]
   + 1.171652093844782 gammas[1] + 1.579628363899807 gammas[2]
   - 1.017636964542434 gammas[3] + tau[33,4] = 0
 tau_computation_33_5: - betas[5] - 0.1119083925633281 gammas[0]
   + 1.171652093844782 gammas[1] + 1.579628363899807 gammas[2]
   - 1.017636964542434 gammas[3] + tau[33,5] = 0
 tau_computation_33_6: - betas[6] - 0.1119083925633281 gammas[0]
   + 1.171652093844782 gammas[1] + 1.579628363899807 gammas[2]
   - 1.017636964542434 gammas[3] + tau[33,6] = 0
 tau_computation_33_7: - betas[7] - 0.1119083925633281 gammas[0]
   + 1.171652093844782 gammas[1] + 1.579628363899807 gammas[2]
   - 1.017636964542434 gammas[3] + tau[33,7] = 0
 Ensure_only_one_R_true_33: R[33,0] + R[33,1] + R[33,2] + R[33,3] + R[33,4]
   + R[33,5] + R[33,6] + R[33,7] = 1
 Define_Q_hat_33: - 1410.147544236772 R[33,0] - 665.574566352488 R[33,1]
   - 684.6367376303186 R[33,2] - 670.110852724704 R[33,3]
   - 635.8553292341912 R[33,4] - 635.8554953004174 R[33,5]
   - 676.2705905363623 R[33,6] - 665.9527389674838 R[33,7] + Q_hat[33] = 0
 Adjust_Q_hat_33: Q0_var[33] - Q_hat[33] + Q_hat_adjusted[33] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_33: Total_Demand_Up_to_K_minus_1[33]
   - 56.41914746621523 R[33,1] - 153.513225931255 R[33,2]
   - 209.9289053185331 R[33,3] - 265.5945348453552 R[33,4]
   - 322.010211767234 R[33,5] - 403.0358042455908 R[33,6]
   - 553.121104136178 R[33,7] = 0
 Constr_Total_Demand_from_K_to_T_33: Total_Demand_from_k_to_T[33]
   - 665.9524631645834 R[33,0] - 609.5333156983681 R[33,1]
   - 512.4392372333285 R[33,2] - 456.0235578460503 R[33,3]
   - 400.3579283192282 R[33,4] - 343.9422513973494 R[33,5]
   - 262.9166589189926 R[33,6] - 112.8313590284054 R[33,7] = 0
 Constr_Left_0_diff_aux_33: Sold_0[33] - Q0_var[33] + Left_0_aux_33 = 0
 Constr_Lost_0_diff_aux_33: - Total_Demand_Up_to_K_minus_1[33] + Q0_var[33]
   + Lost_0_aux_33 = 0
 Constr_Q1_plus_left_33: - Left_0[33] + Q1_plus_left[33] - Q1_var[33] = 0
 Constr_Left_1_diff_aux_33: Sold_1[33] - Q1_plus_left[33] + Left_0_aux_33
   = 0
 Constr_Lost_1_diff_aux_33: - Total_Demand_from_k_to_T[33]
   + Q1_plus_left[33] + Lost_0_aux_33 = 0
 Profit_Constraint_33: - 700 Sold_0[33] + 700 Lost_0[33] - 700 Sold_1[33]
   + 300 Left_1[33] + 700 Lost_1[33] + profits_vars[33] = 0
 R714: 1.906869221658971 alphas[0] + 0.0624273002039462 alphas[1]
   + 0.9191162298511577 alphas[2] - 1.314168571815939 alphas[3] - alphas[4]
   + f_var[34] = 0
 Q0_upper_bound_34:
   - 2028.416054007841 Fraction_for_second_order_amount[34] + Q0_var[34]
   = 0
 tau_computation_34_0: - betas[0] + 1.906869221658971 gammas[0]
   + 0.0624273002039462 gammas[1] + 0.9191162298511577 gammas[2]
   - 1.314168571815939 gammas[3] + tau[34,0] = 0
 tau_computation_34_1: - betas[1] + 1.906869221658971 gammas[0]
   + 0.0624273002039462 gammas[1] + 0.9191162298511577 gammas[2]
   - 1.314168571815939 gammas[3] + tau[34,1] = 0
 tau_computation_34_2: - betas[2] + 1.906869221658971 gammas[0]
   + 0.0624273002039462 gammas[1] + 0.9191162298511577 gammas[2]
   - 1.314168571815939 gammas[3] + tau[34,2] = 0
 tau_computation_34_3: - betas[3] + 1.906869221658971 gammas[0]
   + 0.0624273002039462 gammas[1] + 0.9191162298511577 gammas[2]
   - 1.314168571815939 gammas[3] + tau[34,3] = 0
 tau_computation_34_4: - betas[4] + 1.906869221658971 gammas[0]
   + 0.0624273002039462 gammas[1] + 0.9191162298511577 gammas[2]
   - 1.314168571815939 gammas[3] + tau[34,4] = 0
 tau_computation_34_5: - betas[5] + 1.906869221658971 gammas[0]
   + 0.0624273002039462 gammas[1] + 0.9191162298511577 gammas[2]
   - 1.314168571815939 gammas[3] + tau[34,5] = 0
 tau_computation_34_6: - betas[6] + 1.906869221658971 gammas[0]
   + 0.0624273002039462 gammas[1] + 0.9191162298511577 gammas[2]
   - 1.314168571815939 gammas[3] + tau[34,6] = 0
 tau_computation_34_7: - betas[7] + 1.906869221658971 gammas[0]
   + 0.0624273002039462 gammas[1] + 0.9191162298511577 gammas[2]
   - 1.314168571815939 gammas[3] + tau[34,7] = 0
 Ensure_only_one_R_true_34: R[34,0] + R[34,1] + R[34,2] + R[34,3] + R[34,4]
   + R[34,5] + R[34,6] + R[34,7] = 1
 Define_Q_hat_34: - 1465.498096992268 R[34,0] - 1089.63612352881 R[34,1]
   - 1065.290757042311 R[34,2] - 1061.928460399658 R[34,3]
   - 1017.733480158229 R[34,4] - 1017.73348164558 R[34,5]
   - 1098.92818545655 R[34,6] - 1145.627890748708 R[34,7] + Q_hat[34] = 0
 Adjust_Q_hat_34: Q0_var[34] - Q_hat[34] + Q_hat_adjusted[34] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_34: Total_Demand_Up_to_K_minus_1[34]
   - 17.85856980704314 R[34,0] - 119.7270878668803 R[34,1]
   - 164.1997100395838 R[34,2] - 266.0672288067818 R[34,3]
   - 393.2254974917866 R[34,4] - 495.093016352064 R[34,5]
   - 709.5535378860424 R[34,6] - 941.8925750063599 R[34,7] = 0
 Constr_Total_Demand_from_K_to_T_34: Total_Demand_from_k_to_T[34]
   - 1127.769042919887 R[34,0] - 1025.90052486005 R[34,1]
   - 981.4279026873464 R[34,2] - 879.5603839201485 R[34,3]
   - 752.4021152351438 R[34,4] - 650.5345963748664 R[34,5]
   - 436.074074840888 R[34,6] - 203.7350377205705 R[34,7] = 0
 Constr_Left_0_diff_aux_34: Sold_0[34] - Q0_var[34] + Left_0_aux_34 = 0
 Constr_Lost_0_diff_aux_34: - Total_Demand_Up_to_K_minus_1[34] + Q0_var[34]
   + Lost_0_aux_34 = 0
 Constr_Q1_plus_left_34: - Left_0[34] + Q1_plus_left[34] - Q1_var[34] = 0
 Constr_Left_1_diff_aux_34: Sold_1[34] - Q1_plus_left[34] + Left_0_aux_34
   = 0
 Constr_Lost_1_diff_aux_34: - Total_Demand_from_k_to_T[34]
   + Q1_plus_left[34] + Lost_0_aux_34 = 0
 Profit_Constraint_34: - 700 Sold_0[34] + 700 Lost_0[34] - 700 Sold_1[34]
   + 300 Left_1[34] + 700 Lost_1[34] + profits_vars[34] = 0
 R735: - 0.2926198299344603 alphas[0] - 0.022935205567689 alphas[1]
   + 0.1299655318483328 alphas[2] - 1.309494249403513 alphas[3] - alphas[4]
   + f_var[35] = 0
 Q0_upper_bound_35:
   - 2028.416054007841 Fraction_for_second_order_amount[35] + Q0_var[35]
   = 0
 tau_computation_35_0: - betas[0] - 0.2926198299344603 gammas[0]
   - 0.022935205567689 gammas[1] + 0.1299655318483328 gammas[2]
   - 1.309494249403513 gammas[3] + tau[35,0] = 0
 tau_computation_35_1: - betas[1] - 0.2926198299344603 gammas[0]
   - 0.022935205567689 gammas[1] + 0.1299655318483328 gammas[2]
   - 1.309494249403513 gammas[3] + tau[35,1] = 0
 tau_computation_35_2: - betas[2] - 0.2926198299344603 gammas[0]
   - 0.022935205567689 gammas[1] + 0.1299655318483328 gammas[2]
   - 1.309494249403513 gammas[3] + tau[35,2] = 0
 tau_computation_35_3: - betas[3] - 0.2926198299344603 gammas[0]
   - 0.022935205567689 gammas[1] + 0.1299655318483328 gammas[2]
   - 1.309494249403513 gammas[3] + tau[35,3] = 0
 tau_computation_35_4: - betas[4] - 0.2926198299344603 gammas[0]
   - 0.022935205567689 gammas[1] + 0.1299655318483328 gammas[2]
   - 1.309494249403513 gammas[3] + tau[35,4] = 0
 tau_computation_35_5: - betas[5] - 0.2926198299344603 gammas[0]
   - 0.022935205567689 gammas[1] + 0.1299655318483328 gammas[2]
   - 1.309494249403513 gammas[3] + tau[35,5] = 0
 tau_computation_35_6: - betas[6] - 0.2926198299344603 gammas[0]
   - 0.022935205567689 gammas[1] + 0.1299655318483328 gammas[2]
   - 1.309494249403513 gammas[3] + tau[35,6] = 0
 tau_computation_35_7: - betas[7] - 0.2926198299344603 gammas[0]
   - 0.022935205567689 gammas[1] + 0.1299655318483328 gammas[2]
   - 1.309494249403513 gammas[3] + tau[35,7] = 0
 Ensure_only_one_R_true_35: R[35,0] + R[35,1] + R[35,2] + R[35,3] + R[35,4]
   + R[35,5] + R[35,6] + R[35,7] = 1
 Define_Q_hat_35: - 1616.977433666191 R[35,0] - 1611.567782786439 R[35,1]
   - 1667.86587330821 R[35,2] - 1651.893116747177 R[35,3]
   - 1515.097888859804 R[35,4] - 1515.097962447129 R[35,5]
   - 1412.012096739046 R[35,6] - 1371.509424346211 R[35,7] + Q_hat[35] = 0
 Adjust_Q_hat_35: Q0_var[35] - Q_hat[35] + Q_hat_adjusted[35] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_35: Total_Demand_Up_to_K_minus_1[35]
   - 66.73259497289331 R[35,0] - 222.9045032742931 R[35,1]
   - 480.7404694511511 R[35,2] - 636.9118373461349 R[35,3]
   - 700.7923452938296 R[35,4] - 856.9637124817061 R[35,5]
   - 856.9637124817061 R[35,6] - 1059.166411412334 R[35,7] = 0
 Constr_Total_Demand_from_K_to_T_35: Total_Demand_from_k_to_T[35]
   - 1304.776550815194 R[35,0] - 1148.604642513794 R[35,1]
   - 890.7686763369359 R[35,2] - 734.597308441952 R[35,3]
   - 670.7168004942573 R[35,4] - 514.5454333063808 R[35,5]
   - 514.5454333063808 R[35,6] - 312.3427343757531 R[35,7] = 0
 Constr_Left_0_diff_aux_35: Sold_0[35] - Q0_var[35] + Left_0_aux_35 = 0
 Constr_Lost_0_diff_aux_35: - Total_Demand_Up_to_K_minus_1[35] + Q0_var[35]
   + Lost_0_aux_35 = 0
 Constr_Q1_plus_left_35: - Left_0[35] + Q1_plus_left[35] - Q1_var[35] = 0
 Constr_Left_1_diff_aux_35: Sold_1[35] - Q1_plus_left[35] + Left_0_aux_35
   = 0
 Constr_Lost_1_diff_aux_35: - Total_Demand_from_k_to_T[35]
   + Q1_plus_left[35] + Lost_0_aux_35 = 0
 Profit_Constraint_35: - 700 Sold_0[35] + 700 Lost_0[35] - 700 Sold_1[35]
   + 300 Left_1[35] + 700 Lost_1[35] + profits_vars[35] = 0
 R756: - 0.272272897315425 alphas[0] + 0.0446249254683583 alphas[1]
   + 0.3134899696743634 alphas[2] + 0.570584013404363 alphas[3] - alphas[4]
   + f_var[36] = 0
 Q0_upper_bound_36:
   - 2028.416054007841 Fraction_for_second_order_amount[36] + Q0_var[36]
   = 0
 tau_computation_36_0: - betas[0] - 0.272272897315425 gammas[0]
   + 0.0446249254683583 gammas[1] + 0.3134899696743634 gammas[2]
   + 0.570584013404363 gammas[3] + tau[36,0] = 0
 tau_computation_36_1: - betas[1] - 0.272272897315425 gammas[0]
   + 0.0446249254683583 gammas[1] + 0.3134899696743634 gammas[2]
   + 0.570584013404363 gammas[3] + tau[36,1] = 0
 tau_computation_36_2: - betas[2] - 0.272272897315425 gammas[0]
   + 0.0446249254683583 gammas[1] + 0.3134899696743634 gammas[2]
   + 0.570584013404363 gammas[3] + tau[36,2] = 0
 tau_computation_36_3: - betas[3] - 0.272272897315425 gammas[0]
   + 0.0446249254683583 gammas[1] + 0.3134899696743634 gammas[2]
   + 0.570584013404363 gammas[3] + tau[36,3] = 0
 tau_computation_36_4: - betas[4] - 0.272272897315425 gammas[0]
   + 0.0446249254683583 gammas[1] + 0.3134899696743634 gammas[2]
   + 0.570584013404363 gammas[3] + tau[36,4] = 0
 tau_computation_36_5: - betas[5] - 0.272272897315425 gammas[0]
   + 0.0446249254683583 gammas[1] + 0.3134899696743634 gammas[2]
   + 0.570584013404363 gammas[3] + tau[36,5] = 0
 tau_computation_36_6: - betas[6] - 0.272272897315425 gammas[0]
   + 0.0446249254683583 gammas[1] + 0.3134899696743634 gammas[2]
   + 0.570584013404363 gammas[3] + tau[36,6] = 0
 tau_computation_36_7: - betas[7] - 0.272272897315425 gammas[0]
   + 0.0446249254683583 gammas[1] + 0.3134899696743634 gammas[2]
   + 0.570584013404363 gammas[3] + tau[36,7] = 0
 Ensure_only_one_R_true_36: R[36,0] + R[36,1] + R[36,2] + R[36,3] + R[36,4]
   + R[36,5] + R[36,6] + R[36,7] = 1
 Define_Q_hat_36: - 1556.244049506663 R[36,0] - 1485.611032405871 R[36,1]
   - 1455.698991286873 R[36,2] - 1455.677156177923 R[36,3]
   - 1444.837791012756 R[36,4] - 1444.8378049126 R[36,5]
   - 1351.403120475179 R[36,6] - 1286.377276368151 R[36,7] + Q_hat[36] = 0
 Adjust_Q_hat_36: Q0_var[36] - Q_hat[36] + Q_hat_adjusted[36] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_36: Total_Demand_Up_to_K_minus_1[36]
   - 47.13728243063132 R[36,0] - 190.6785949638538 R[36,1]
   - 260.5128782302014 R[36,2] - 404.055373250211 R[36,3]
   - 603.9256585052015 R[36,4] - 747.4681535887578 R[36,5]
   - 879.3025934138514 R[36,6] - 999.2920081744734 R[36,7] = 0
 Constr_Total_Demand_from_K_to_T_36: Total_Demand_from_k_to_T[36]
   - 1239.239715910955 R[36,0] - 1095.698403377732 R[36,1]
   - 1025.864120111385 R[36,2] - 882.3216250913749 R[36,3]
   - 682.4513398363847 R[36,4] - 538.9088447528284 R[36,5]
   - 407.0744049277347 R[36,6] - 287.0849901671128 R[36,7] = 0
 Constr_Left_0_diff_aux_36: Sold_0[36] - Q0_var[36] + Left_0_aux_36 = 0
 Constr_Lost_0_diff_aux_36: - Total_Demand_Up_to_K_minus_1[36] + Q0_var[36]
   + Lost_0_aux_36 = 0
 Constr_Q1_plus_left_36: - Left_0[36] + Q1_plus_left[36] - Q1_var[36] = 0
 Constr_Left_1_diff_aux_36: Sold_1[36] - Q1_plus_left[36] + Left_0_aux_36
   = 0
 Constr_Lost_1_diff_aux_36: - Total_Demand_from_k_to_T[36]
   + Q1_plus_left[36] + Lost_0_aux_36 = 0
 Profit_Constraint_36: - 700 Sold_0[36] + 700 Lost_0[36] - 700 Sold_1[36]
   + 300 Left_1[36] + 700 Lost_1[36] + profits_vars[36] = 0
 R777: - 0.290043303091715 alphas[0] - 0.2774080051893102 alphas[1]
   - 0.0113229613372425 alphas[2] + 0.1553920060914088 alphas[3]
   - alphas[4] + f_var[37] = 0
 Q0_upper_bound_37:
   - 2028.416054007841 Fraction_for_second_order_amount[37] + Q0_var[37]
   = 0
 tau_computation_37_0: - betas[0] - 0.290043303091715 gammas[0]
   - 0.2774080051893102 gammas[1] - 0.0113229613372425 gammas[2]
   + 0.1553920060914088 gammas[3] + tau[37,0] = 0
 tau_computation_37_1: - betas[1] - 0.290043303091715 gammas[0]
   - 0.2774080051893102 gammas[1] - 0.0113229613372425 gammas[2]
   + 0.1553920060914088 gammas[3] + tau[37,1] = 0
 tau_computation_37_2: - betas[2] - 0.290043303091715 gammas[0]
   - 0.2774080051893102 gammas[1] - 0.0113229613372425 gammas[2]
   + 0.1553920060914088 gammas[3] + tau[37,2] = 0
 tau_computation_37_3: - betas[3] - 0.290043303091715 gammas[0]
   - 0.2774080051893102 gammas[1] - 0.0113229613372425 gammas[2]
   + 0.1553920060914088 gammas[3] + tau[37,3] = 0
 tau_computation_37_4: - betas[4] - 0.290043303091715 gammas[0]
   - 0.2774080051893102 gammas[1] - 0.0113229613372425 gammas[2]
   + 0.1553920060914088 gammas[3] + tau[37,4] = 0
 tau_computation_37_5: - betas[5] - 0.290043303091715 gammas[0]
   - 0.2774080051893102 gammas[1] - 0.0113229613372425 gammas[2]
   + 0.1553920060914088 gammas[3] + tau[37,5] = 0
 tau_computation_37_6: - betas[6] - 0.290043303091715 gammas[0]
   - 0.2774080051893102 gammas[1] - 0.0113229613372425 gammas[2]
   + 0.1553920060914088 gammas[3] + tau[37,6] = 0
 tau_computation_37_7: - betas[7] - 0.290043303091715 gammas[0]
   - 0.2774080051893102 gammas[1] - 0.0113229613372425 gammas[2]
   + 0.1553920060914088 gammas[3] + tau[37,7] = 0
 Ensure_only_one_R_true_37: R[37,0] + R[37,1] + R[37,2] + R[37,3] + R[37,4]
   + R[37,5] + R[37,6] + R[37,7] = 1
 Define_Q_hat_37: - 1619.318346049584 R[37,0] - 1700.578417764631 R[37,1]
   - 1703.105642220724 R[37,2] - 1698.37347700026 R[37,3]
   - 1651.764813681873 R[37,4] - 1651.764814899768 R[37,5]
   - 1544.477327154716 R[37,6] - 1550.294317752225 R[37,7] + Q_hat[37] = 0
 Adjust_Q_hat_37: Q0_var[37] - Q_hat[37] + Q_hat_adjusted[37] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_37: Total_Demand_Up_to_K_minus_1[37]
   - 67.4878782560732 R[37,0] - 233.3816321506545 R[37,1]
   - 389.6173647409072 R[37,2] - 555.5112205935731 R[37,3]
   - 731.5458824229115 R[37,4] - 897.4397385141142 R[37,5]
   - 990.0898045812996 R[37,6] - 1218.506327701144 R[37,7] = 0
 Constr_Total_Demand_from_K_to_T_37: Total_Demand_from_k_to_T[37]
   - 1482.806161627476 R[37,0] - 1316.912407732895 R[37,1]
   - 1160.676675142642 R[37,2] - 994.7828192899765 R[37,3]
   - 818.7481574606379 R[37,4] - 652.8543013694352 R[37,5]
   - 560.2042353022498 R[37,6] - 331.7877121824054 R[37,7] = 0
 Constr_Left_0_diff_aux_37: Sold_0[37] - Q0_var[37] + Left_0_aux_37 = 0
 Constr_Lost_0_diff_aux_37: - Total_Demand_Up_to_K_minus_1[37] + Q0_var[37]
   + Lost_0_aux_37 = 0
 Constr_Q1_plus_left_37: - Left_0[37] + Q1_plus_left[37] - Q1_var[37] = 0
 Constr_Left_1_diff_aux_37: Sold_1[37] - Q1_plus_left[37] + Left_0_aux_37
   = 0
 Constr_Lost_1_diff_aux_37: - Total_Demand_from_k_to_T[37]
   + Q1_plus_left[37] + Lost_0_aux_37 = 0
 Profit_Constraint_37: - 700 Sold_0[37] + 700 Lost_0[37] - 700 Sold_1[37]
   + 300 Left_1[37] + 700 Lost_1[37] + profits_vars[37] = 0
 R798: - 1.490392572539918 alphas[0] - 0.2729269321273108 alphas[1]
   + 0.6642276067877726 alphas[2] - 0.6205674296544516 alphas[3]
   - alphas[4] + f_var[38] = 0
 Q0_upper_bound_38:
   - 2028.416054007841 Fraction_for_second_order_amount[38] + Q0_var[38]
   = 0
 tau_computation_38_0: - betas[0] - 1.490392572539918 gammas[0]
   - 0.2729269321273108 gammas[1] + 0.6642276067877726 gammas[2]
   - 0.6205674296544516 gammas[3] + tau[38,0] = 0
 tau_computation_38_1: - betas[1] - 1.490392572539918 gammas[0]
   - 0.2729269321273108 gammas[1] + 0.6642276067877726 gammas[2]
   - 0.6205674296544516 gammas[3] + tau[38,1] = 0
 tau_computation_38_2: - betas[2] - 1.490392572539918 gammas[0]
   - 0.2729269321273108 gammas[1] + 0.6642276067877726 gammas[2]
   - 0.6205674296544516 gammas[3] + tau[38,2] = 0
 tau_computation_38_3: - betas[3] - 1.490392572539918 gammas[0]
   - 0.2729269321273108 gammas[1] + 0.6642276067877726 gammas[2]
   - 0.6205674296544516 gammas[3] + tau[38,3] = 0
 tau_computation_38_4: - betas[4] - 1.490392572539918 gammas[0]
   - 0.2729269321273108 gammas[1] + 0.6642276067877726 gammas[2]
   - 0.6205674296544516 gammas[3] + tau[38,4] = 0
 tau_computation_38_5: - betas[5] - 1.490392572539918 gammas[0]
   - 0.2729269321273108 gammas[1] + 0.6642276067877726 gammas[2]
   - 0.6205674296544516 gammas[3] + tau[38,5] = 0
 tau_computation_38_6: - betas[6] - 1.490392572539918 gammas[0]
   - 0.2729269321273108 gammas[1] + 0.6642276067877726 gammas[2]
   - 0.6205674296544516 gammas[3] + tau[38,6] = 0
 tau_computation_38_7: - betas[7] - 1.490392572539918 gammas[0]
   - 0.2729269321273108 gammas[1] + 0.6642276067877726 gammas[2]
   - 0.6205674296544516 gammas[3] + tau[38,7] = 0
 Ensure_only_one_R_true_38: R[38,0] + R[38,1] + R[38,2] + R[38,3] + R[38,4]
   + R[38,5] + R[38,6] + R[38,7] = 1
 Define_Q_hat_38: - 2529.11630110872 R[38,0] - 1439.94004393726 R[38,1]
   - 1462.420102308449 R[38,2] - 1457.354558826375 R[38,3]
   - 1438.136134821375 R[38,4] - 1438.135999792679 R[38,5]
   - 1548.880097889291 R[38,6] - 1563.231721339206 R[38,7] + Q_hat[38] = 0
 Adjust_Q_hat_38: Q0_var[38] - Q_hat[38] + Q_hat_adjusted[38] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_38: Total_Demand_Up_to_K_minus_1[38]
   - 361.0294868195984 R[38,0] - 480.4720672264529 R[38,1]
   - 762.381959788152 R[38,2] - 881.7891359365061 R[38,3]
   - 881.7891359365061 R[38,4] - 1001.196312510189 R[38,5]
   - 1173.762062349642 R[38,6] - 1324.417089681451 R[38,7] = 0
 Constr_Total_Demand_from_K_to_T_38: Total_Demand_from_k_to_T[38]
   - 1202.201956009216 R[38,0] - 1082.759375602362 R[38,1]
   - 800.8494830406627 R[38,2] - 681.4423068923086 R[38,3]
   - 681.4423068923086 R[38,4] - 562.0351303186255 R[38,5]
   - 389.4693804791731 R[38,6] - 238.8143531473638 R[38,7] = 0
 Constr_Left_0_diff_aux_38: Sold_0[38] - Q0_var[38] + Left_0_aux_38 = 0
 Constr_Lost_0_diff_aux_38: - Total_Demand_Up_to_K_minus_1[38] + Q0_var[38]
   + Lost_0_aux_38 = 0
 Constr_Q1_plus_left_38: - Left_0[38] + Q1_plus_left[38] - Q1_var[38] = 0
 Constr_Left_1_diff_aux_38: Sold_1[38] - Q1_plus_left[38] + Left_0_aux_38
   = 0
 Constr_Lost_1_diff_aux_38: - Total_Demand_from_k_to_T[38]
   + Q1_plus_left[38] + Lost_0_aux_38 = 0
 Profit_Constraint_38: - 700 Sold_0[38] + 700 Lost_0[38] - 700 Sold_1[38]
   + 300 Left_1[38] + 700 Lost_1[38] + profits_vars[38] = 0
 R819: - 0.5283629773439216 alphas[0] + 0.6280844091501678 alphas[1]
   - 0.4589450445991258 alphas[2] + 0.5219497780887212 alphas[3]
   - alphas[4] + f_var[39] = 0
 Q0_upper_bound_39:
   - 2028.416054007841 Fraction_for_second_order_amount[39] + Q0_var[39]
   = 0
 tau_computation_39_0: - betas[0] - 0.5283629773439216 gammas[0]
   + 0.6280844091501678 gammas[1] - 0.4589450445991258 gammas[2]
   + 0.5219497780887212 gammas[3] + tau[39,0] = 0
 tau_computation_39_1: - betas[1] - 0.5283629773439216 gammas[0]
   + 0.6280844091501678 gammas[1] - 0.4589450445991258 gammas[2]
   + 0.5219497780887212 gammas[3] + tau[39,1] = 0
 tau_computation_39_2: - betas[2] - 0.5283629773439216 gammas[0]
   + 0.6280844091501678 gammas[1] - 0.4589450445991258 gammas[2]
   + 0.5219497780887212 gammas[3] + tau[39,2] = 0
 tau_computation_39_3: - betas[3] - 0.5283629773439216 gammas[0]
   + 0.6280844091501678 gammas[1] - 0.4589450445991258 gammas[2]
   + 0.5219497780887212 gammas[3] + tau[39,3] = 0
 tau_computation_39_4: - betas[4] - 0.5283629773439216 gammas[0]
   + 0.6280844091501678 gammas[1] - 0.4589450445991258 gammas[2]
   + 0.5219497780887212 gammas[3] + tau[39,4] = 0
 tau_computation_39_5: - betas[5] - 0.5283629773439216 gammas[0]
   + 0.6280844091501678 gammas[1] - 0.4589450445991258 gammas[2]
   + 0.5219497780887212 gammas[3] + tau[39,5] = 0
 tau_computation_39_6: - betas[6] - 0.5283629773439216 gammas[0]
   + 0.6280844091501678 gammas[1] - 0.4589450445991258 gammas[2]
   + 0.5219497780887212 gammas[3] + tau[39,6] = 0
 tau_computation_39_7: - betas[7] - 0.5283629773439216 gammas[0]
   + 0.6280844091501678 gammas[1] - 0.4589450445991258 gammas[2]
   + 0.5219497780887212 gammas[3] + tau[39,7] = 0
 Ensure_only_one_R_true_39: R[39,0] + R[39,1] + R[39,2] + R[39,3] + R[39,4]
   + R[39,5] + R[39,6] + R[39,7] = 1
 Define_Q_hat_39: - 1823.351578152731 R[39,0] - 2017.748213247011 R[39,1]
   - 1998.375136919483 R[39,2] - 1999.764792862962 R[39,3]
   - 1882.829277315468 R[39,4] - 1882.829298241096 R[39,5]
   - 1958.730294147103 R[39,6] - 1994.024537244877 R[39,7] + Q_hat[39] = 0
 Adjust_Q_hat_39: Q0_var[39] - Q_hat[39] + Q_hat_adjusted[39] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_39: Total_Demand_Up_to_K_minus_1[39]
   - 133.3181461951562 R[39,0] - 330.0142466243368 R[39,1]
   - 486.473409945026 R[39,2] - 683.169496164909 R[39,3]
   - 832.3703838900963 R[39,4] - 1029.066470504638 R[39,5]
   - 1308.774901227405 R[39,6] - 1600.632086988877 R[39,7] = 0
 Constr_Total_Demand_from_K_to_T_39: Total_Demand_from_k_to_T[39]
   - 1860.706114022805 R[39,0] - 1664.010013593625 R[39,1]
   - 1507.550850272936 R[39,2] - 1310.854764053052 R[39,3]
   - 1161.653876327865 R[39,4] - 964.957789713323 R[39,5]
   - 685.2493589905565 R[39,6] - 393.3921732290844 R[39,7] = 0
 Constr_Left_0_diff_aux_39: Sold_0[39] - Q0_var[39] + Left_0_aux_39 = 0
 Constr_Lost_0_diff_aux_39: - Total_Demand_Up_to_K_minus_1[39] + Q0_var[39]
   + Lost_0_aux_39 = 0
 Constr_Q1_plus_left_39: - Left_0[39] + Q1_plus_left[39] - Q1_var[39] = 0
 Constr_Left_1_diff_aux_39: Sold_1[39] - Q1_plus_left[39] + Left_0_aux_39
   = 0
 Constr_Lost_1_diff_aux_39: - Total_Demand_from_k_to_T[39]
   + Q1_plus_left[39] + Lost_0_aux_39 = 0
 Profit_Constraint_39: - 700 Sold_0[39] + 700 Lost_0[39] - 700 Sold_1[39]
   + 300 Left_1[39] + 700 Lost_1[39] + profits_vars[39] = 0
 R840: 0.6554521452111429 alphas[0] - 1.270939366109456 alphas[1]
   - 1.465554081238291 alphas[2] + 0.1661368510093704 alphas[3] - alphas[4]
   + f_var[40] = 0
 Q0_upper_bound_40:
   - 2028.416054007841 Fraction_for_second_order_amount[40] + Q0_var[40]
   = 0
 tau_computation_40_0: - betas[0] + 0.6554521452111429 gammas[0]
   - 1.270939366109456 gammas[1] - 1.465554081238291 gammas[2]
   + 0.1661368510093704 gammas[3] + tau[40,0] = 0
 tau_computation_40_1: - betas[1] + 0.6554521452111429 gammas[0]
   - 1.270939366109456 gammas[1] - 1.465554081238291 gammas[2]
   + 0.1661368510093704 gammas[3] + tau[40,1] = 0
 tau_computation_40_2: - betas[2] + 0.6554521452111429 gammas[0]
   - 1.270939366109456 gammas[1] - 1.465554081238291 gammas[2]
   + 0.1661368510093704 gammas[3] + tau[40,2] = 0
 tau_computation_40_3: - betas[3] + 0.6554521452111429 gammas[0]
   - 1.270939366109456 gammas[1] - 1.465554081238291 gammas[2]
   + 0.1661368510093704 gammas[3] + tau[40,3] = 0
 tau_computation_40_4: - betas[4] + 0.6554521452111429 gammas[0]
   - 1.270939366109456 gammas[1] - 1.465554081238291 gammas[2]
   + 0.1661368510093704 gammas[3] + tau[40,4] = 0
 tau_computation_40_5: - betas[5] + 0.6554521452111429 gammas[0]
   - 1.270939366109456 gammas[1] - 1.465554081238291 gammas[2]
   + 0.1661368510093704 gammas[3] + tau[40,5] = 0
 tau_computation_40_6: - betas[6] + 0.6554521452111429 gammas[0]
   - 1.270939366109456 gammas[1] - 1.465554081238291 gammas[2]
   + 0.1661368510093704 gammas[3] + tau[40,6] = 0
 tau_computation_40_7: - betas[7] + 0.6554521452111429 gammas[0]
   - 1.270939366109456 gammas[1] - 1.465554081238291 gammas[2]
   + 0.1661368510093704 gammas[3] + tau[40,7] = 0
 Ensure_only_one_R_true_40: R[40,0] + R[40,1] + R[40,2] + R[40,3] + R[40,4]
   + R[40,5] + R[40,6] + R[40,7] = 1
 Define_Q_hat_40: - 2025.290479561419 R[40,0] - 2685.067158262012 R[40,1]
   - 2755.685772467925 R[40,2] - 2743.681298445248 R[40,3]
   - 2750.281269936589 R[40,4] - 2750.281275912186 R[40,5]
   - 2725.377190251825 R[40,6] - 2623.131233296414 R[40,7] + Q_hat[40] = 0
 Adjust_Q_hat_40: Q0_var[40] - Q_hat[40] + Q_hat_adjusted[40] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_40: Total_Demand_Up_to_K_minus_1[40]
   - 198.4726891587142 R[40,0] - 464.4365905590828 R[40,1]
   - 868.3424489778176 R[40,2] - 1134.306350534185 R[40,3]
   - 1392.490551448338 R[40,4] - 1658.454452929146 R[40,5]
   - 1818.317888513554 R[40,6] - 2091.203152144836 R[40,7] = 0
 Constr_Total_Demand_from_K_to_T_40: Total_Demand_from_k_to_T[40]
   - 2424.658265947738 R[40,0] - 2158.694364547369 R[40,1]
   - 1754.788506128634 R[40,2] - 1488.824604572267 R[40,3]
   - 1230.640403658113 R[40,4] - 964.6765021773056 R[40,5]
   - 804.8130665928982 R[40,6] - 531.9278029616157 R[40,7] = 0
 Constr_Left_0_diff_aux_40: Sold_0[40] - Q0_var[40] + Left_0_aux_40 = 0
 Constr_Lost_0_diff_aux_40: - Total_Demand_Up_to_K_minus_1[40] + Q0_var[40]
   + Lost_0_aux_40 = 0
 Constr_Q1_plus_left_40: - Left_0[40] + Q1_plus_left[40] - Q1_var[40] = 0
 Constr_Left_1_diff_aux_40: Sold_1[40] - Q1_plus_left[40] + Left_0_aux_40
   = 0
 Constr_Lost_1_diff_aux_40: - Total_Demand_from_k_to_T[40]
   + Q1_plus_left[40] + Lost_0_aux_40 = 0
 Profit_Constraint_40: - 700 Sold_0[40] + 700 Lost_0[40] - 700 Sold_1[40]
   + 300 Left_1[40] + 700 Lost_1[40] + profits_vars[40] = 0
 R861: 0.3707155139526951 alphas[0] + 0.88879779138409 alphas[1]
   + 1.245864751663434 alphas[2] + 0.7969559274606892 alphas[3] - alphas[4]
   + f_var[41] = 0
 Q0_upper_bound_41:
   - 2028.416054007841 Fraction_for_second_order_amount[41] + Q0_var[41]
   = 0
 tau_computation_41_0: - betas[0] + 0.3707155139526951 gammas[0]
   + 0.88879779138409 gammas[1] + 1.245864751663434 gammas[2]
   + 0.7969559274606892 gammas[3] + tau[41,0] = 0
 tau_computation_41_1: - betas[1] + 0.3707155139526951 gammas[0]
   + 0.88879779138409 gammas[1] + 1.245864751663434 gammas[2]
   + 0.7969559274606892 gammas[3] + tau[41,1] = 0
 tau_computation_41_2: - betas[2] + 0.3707155139526951 gammas[0]
   + 0.88879779138409 gammas[1] + 1.245864751663434 gammas[2]
   + 0.7969559274606892 gammas[3] + tau[41,2] = 0
 tau_computation_41_3: - betas[3] + 0.3707155139526951 gammas[0]
   + 0.88879779138409 gammas[1] + 1.245864751663434 gammas[2]
   + 0.7969559274606892 gammas[3] + tau[41,3] = 0
 tau_computation_41_4: - betas[4] + 0.3707155139526951 gammas[0]
   + 0.88879779138409 gammas[1] + 1.245864751663434 gammas[2]
   + 0.7969559274606892 gammas[3] + tau[41,4] = 0
 tau_computation_41_5: - betas[5] + 0.3707155139526951 gammas[0]
   + 0.88879779138409 gammas[1] + 1.245864751663434 gammas[2]
   + 0.7969559274606892 gammas[3] + tau[41,5] = 0
 tau_computation_41_6: - betas[6] + 0.3707155139526951 gammas[0]
   + 0.88879779138409 gammas[1] + 1.245864751663434 gammas[2]
   + 0.7969559274606892 gammas[3] + tau[41,6] = 0
 tau_computation_41_7: - betas[7] + 0.3707155139526951 gammas[0]
   + 0.88879779138409 gammas[1] + 1.245864751663434 gammas[2]
   + 0.7969559274606892 gammas[3] + tau[41,7] = 0
 Ensure_only_one_R_true_41: R[41,0] + R[41,1] + R[41,2] + R[41,3] + R[41,4]
   + R[41,5] + R[41,6] + R[41,7] = 1
 Define_Q_hat_41: - 2220.524364179639 R[41,0] - 1019.782046909943 R[41,1]
   - 991.3251506947782 R[41,2] - 986.8349519743522 R[41,3]
   - 948.1807854784126 R[41,4] - 948.1807492381399 R[41,5]
   - 817.5086516453484 R[41,6] - 799.0667530833817 R[41,7] + Q_hat[41] = 0
 Adjust_Q_hat_41: Q0_var[41] - Q_hat[41] + Q_hat_adjusted[41] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_41: Total_Demand_Up_to_K_minus_1[41]
   - 261.4638931048836 R[41,0] - 340.8538496430822 R[41,1]
   - 457.9028179719626 R[41,2] - 537.285782139053 R[41,3]
   - 537.285782139053 R[41,4] - 616.6687461295611 R[41,5]
   - 635.2121709819166 R[41,6] - 640.3005463220409 R[41,7] = 0
 Constr_Total_Demand_from_K_to_T_41: Total_Demand_from_k_to_T[41]
   - 537.6025811981683 R[41,0] - 458.2126246599697 R[41,1]
   - 341.1636563310893 R[41,2] - 261.7806921639989 R[41,3]
   - 261.7806921639989 R[41,4] - 182.3977281734909 R[41,5]
   - 163.8543033211353 R[41,6] - 158.7659279810111 R[41,7] = 0
 Constr_Left_0_diff_aux_41: Sold_0[41] - Q0_var[41] + Left_0_aux_41 = 0
 Constr_Lost_0_diff_aux_41: - Total_Demand_Up_to_K_minus_1[41] + Q0_var[41]
   + Lost_0_aux_41 = 0
 Constr_Q1_plus_left_41: - Left_0[41] + Q1_plus_left[41] - Q1_var[41] = 0
 Constr_Left_1_diff_aux_41: Sold_1[41] - Q1_plus_left[41] + Left_0_aux_41
   = 0
 Constr_Lost_1_diff_aux_41: - Total_Demand_from_k_to_T[41]
   + Q1_plus_left[41] + Lost_0_aux_41 = 0
 Profit_Constraint_41: - 700 Sold_0[41] + 700 Lost_0[41] - 700 Sold_1[41]
   + 300 Left_1[41] + 700 Lost_1[41] + profits_vars[41] = 0
 R882: - 0.5864345217416338 alphas[0] + 0.609930294595618 alphas[1]
   - 0.206793517918046 alphas[2] - 0.1569087301744184 alphas[3] - alphas[4]
   + f_var[42] = 0
 Q0_upper_bound_42:
   - 2028.416054007841 Fraction_for_second_order_amount[42] + Q0_var[42]
   = 0
 tau_computation_42_0: - betas[0] - 0.5864345217416338 gammas[0]
   + 0.609930294595618 gammas[1] - 0.206793517918046 gammas[2]
   - 0.1569087301744184 gammas[3] + tau[42,0] = 0
 tau_computation_42_1: - betas[1] - 0.5864345217416338 gammas[0]
   + 0.609930294595618 gammas[1] - 0.206793517918046 gammas[2]
   - 0.1569087301744184 gammas[3] + tau[42,1] = 0
 tau_computation_42_2: - betas[2] - 0.5864345217416338 gammas[0]
   + 0.609930294595618 gammas[1] - 0.206793517918046 gammas[2]
   - 0.1569087301744184 gammas[3] + tau[42,2] = 0
 tau_computation_42_3: - betas[3] - 0.5864345217416338 gammas[0]
   + 0.609930294595618 gammas[1] - 0.206793517918046 gammas[2]
   - 0.1569087301744184 gammas[3] + tau[42,3] = 0
 tau_computation_42_4: - betas[4] - 0.5864345217416338 gammas[0]
   + 0.609930294595618 gammas[1] - 0.206793517918046 gammas[2]
   - 0.1569087301744184 gammas[3] + tau[42,4] = 0
 tau_computation_42_5: - betas[5] - 0.5864345217416338 gammas[0]
   + 0.609930294595618 gammas[1] - 0.206793517918046 gammas[2]
   - 0.1569087301744184 gammas[3] + tau[42,5] = 0
 tau_computation_42_6: - betas[6] - 0.5864345217416338 gammas[0]
   + 0.609930294595618 gammas[1] - 0.206793517918046 gammas[2]
   - 0.1569087301744184 gammas[3] + tau[42,6] = 0
 tau_computation_42_7: - betas[7] - 0.5864345217416338 gammas[0]
   + 0.609930294595618 gammas[1] - 0.206793517918046 gammas[2]
   - 0.1569087301744184 gammas[3] + tau[42,7] = 0
 Ensure_only_one_R_true_42: R[42,0] + R[42,1] + R[42,2] + R[42,3] + R[42,4]
   + R[42,5] + R[42,6] + R[42,7] = 1
 Define_Q_hat_42: - 2166.82258199939 R[42,0] - 1921.037547755639 R[42,1]
   - 1872.258115841408 R[42,2] - 1877.374256763466 R[42,3]
   - 2004.517804180501 R[42,4] - 2004.517760304843 R[42,5]
   - 2139.12717005228 R[42,6] - 2168.07008365509 R[42,7] + Q_hat[42] = 0
 Adjust_Q_hat_42: Q0_var[42] - Q_hat[42] + Q_hat_adjusted[42] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_42: Total_Demand_Up_to_K_minus_1[42]
   - 244.1372906034592 R[42,0] - 423.4820295847018 R[42,1]
   - 553.4662794086022 R[42,2] - 732.8110560257347 R[42,3]
   - 1013.419951438799 R[42,4] - 1192.764728227327 R[42,5]
   - 1565.892227441055 R[42,6] - 1809.380251590136 R[42,7] = 0
 Constr_Total_Demand_from_K_to_T_42: Total_Demand_from_k_to_T[42]
   - 1923.932514563734 R[42,0] - 1744.587775582491 R[42,1]
   - 1614.603525758591 R[42,2] - 1435.258749141458 R[42,3]
   - 1154.649853728394 R[42,4] - 975.3050769398656 R[42,5]
   - 602.1775777261375 R[42,6] - 358.6895535770571 R[42,7] = 0
 Constr_Left_0_diff_aux_42: Sold_0[42] - Q0_var[42] + Left_0_aux_42 = 0
 Constr_Lost_0_diff_aux_42: - Total_Demand_Up_to_K_minus_1[42] + Q0_var[42]
   + Lost_0_aux_42 = 0
 Constr_Q1_plus_left_42: - Left_0[42] + Q1_plus_left[42] - Q1_var[42] = 0
 Constr_Left_1_diff_aux_42: Sold_1[42] - Q1_plus_left[42] + Left_0_aux_42
   = 0
 Constr_Lost_1_diff_aux_42: - Total_Demand_from_k_to_T[42]
   + Q1_plus_left[42] + Lost_0_aux_42 = 0
 Profit_Constraint_42: - 700 Sold_0[42] + 700 Lost_0[42] - 700 Sold_1[42]
   + 300 Left_1[42] + 700 Lost_1[42] + profits_vars[42] = 0
 R903: 1.754680927335964 alphas[0] - 1.252063613118527 alphas[1]
   + 1.193054073923892 alphas[2] + 1.350259309431124 alphas[3] - alphas[4]
   + f_var[43] = 0
 Q0_upper_bound_43:
   - 2028.416054007841 Fraction_for_second_order_amount[43] + Q0_var[43]
   = 0
 tau_computation_43_0: - betas[0] + 1.754680927335964 gammas[0]
   - 1.252063613118527 gammas[1] + 1.193054073923892 gammas[2]
   + 1.350259309431124 gammas[3] + tau[43,0] = 0
 tau_computation_43_1: - betas[1] + 1.754680927335964 gammas[0]
   - 1.252063613118527 gammas[1] + 1.193054073923892 gammas[2]
   + 1.350259309431124 gammas[3] + tau[43,1] = 0
 tau_computation_43_2: - betas[2] + 1.754680927335964 gammas[0]
   - 1.252063613118527 gammas[1] + 1.193054073923892 gammas[2]
   + 1.350259309431124 gammas[3] + tau[43,2] = 0
 tau_computation_43_3: - betas[3] + 1.754680927335964 gammas[0]
   - 1.252063613118527 gammas[1] + 1.193054073923892 gammas[2]
   + 1.350259309431124 gammas[3] + tau[43,3] = 0
 tau_computation_43_4: - betas[4] + 1.754680927335964 gammas[0]
   - 1.252063613118527 gammas[1] + 1.193054073923892 gammas[2]
   + 1.350259309431124 gammas[3] + tau[43,4] = 0
 tau_computation_43_5: - betas[5] + 1.754680927335964 gammas[0]
   - 1.252063613118527 gammas[1] + 1.193054073923892 gammas[2]
   + 1.350259309431124 gammas[3] + tau[43,5] = 0
 tau_computation_43_6: - betas[6] + 1.754680927335964 gammas[0]
   - 1.252063613118527 gammas[1] + 1.193054073923892 gammas[2]
   + 1.350259309431124 gammas[3] + tau[43,6] = 0
 tau_computation_43_7: - betas[7] + 1.754680927335964 gammas[0]
   - 1.252063613118527 gammas[1] + 1.193054073923892 gammas[2]
   + 1.350259309431124 gammas[3] + tau[43,7] = 0
 Ensure_only_one_R_true_43: R[43,0] + R[43,1] + R[43,2] + R[43,3] + R[43,4]
   + R[43,5] + R[43,6] + R[43,7] = 1
 Define_Q_hat_43: - 1440.080753839313 R[43,0] - 914.333751204439 R[43,1]
   - 970.2721782488636 R[43,2] - 978.892136625826 R[43,3]
   - 953.6259201247658 R[43,4] - 953.6257568508233 R[43,5]
   - 902.0161492145585 R[43,6] - 823.1072762094914 R[43,7] + Q_hat[43] = 0
 Adjust_Q_hat_43: Q0_var[43] - Q_hat[43] + Q_hat_adjusted[43] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_43: Total_Demand_Up_to_K_minus_1[43]
   - 9.657795389997318 R[43,0] - 92.78779959425385 R[43,1]
   - 283.6651549836673 R[43,2] - 366.6821811647664 R[43,3]
   - 448.5839574434233 R[43,4] - 531.6009840297105 R[43,5]
   - 531.6009840297105 R[43,6] - 657.0729437534862 R[43,7] = 0
 Constr_Total_Demand_from_K_to_T_43: Total_Demand_from_k_to_T[43]
   - 813.449201535423 R[43,0] - 730.3191973311664 R[43,1]
   - 539.441841941753 R[43,2] - 456.4248157606539 R[43,3]
   - 374.523039481997 R[43,4] - 291.5060128957097 R[43,5]
   - 291.5060128957097 R[43,6] - 166.034053171934 R[43,7] = 0
 Constr_Left_0_diff_aux_43: Sold_0[43] - Q0_var[43] + Left_0_aux_43 = 0
 Constr_Lost_0_diff_aux_43: - Total_Demand_Up_to_K_minus_1[43] + Q0_var[43]
   + Lost_0_aux_43 = 0
 Constr_Q1_plus_left_43: - Left_0[43] + Q1_plus_left[43] - Q1_var[43] = 0
 Constr_Left_1_diff_aux_43: Sold_1[43] - Q1_plus_left[43] + Left_0_aux_43
   = 0
 Constr_Lost_1_diff_aux_43: - Total_Demand_from_k_to_T[43]
   + Q1_plus_left[43] + Lost_0_aux_43 = 0
 Profit_Constraint_43: - 700 Sold_0[43] + 700 Lost_0[43] - 700 Sold_1[43]
   + 300 Left_1[43] + 700 Lost_1[43] + profits_vars[43] = 0
 R924: - 0.4730729541736584 alphas[0] - 0.9020480301699687 alphas[1]
   - 0.9315130317672309 alphas[2] - 1.390754901918511 alphas[3] - alphas[4]
   + f_var[44] = 0
 Q0_upper_bound_44:
   - 2028.416054007841 Fraction_for_second_order_amount[44] + Q0_var[44]
   = 0
 tau_computation_44_0: - betas[0] - 0.4730729541736584 gammas[0]
   - 0.9020480301699687 gammas[1] - 0.9315130317672309 gammas[2]
   - 1.390754901918511 gammas[3] + tau[44,0] = 0
 tau_computation_44_1: - betas[1] - 0.4730729541736584 gammas[0]
   - 0.9020480301699687 gammas[1] - 0.9315130317672309 gammas[2]
   - 1.390754901918511 gammas[3] + tau[44,1] = 0
 tau_computation_44_2: - betas[2] - 0.4730729541736584 gammas[0]
   - 0.9020480301699687 gammas[1] - 0.9315130317672309 gammas[2]
   - 1.390754901918511 gammas[3] + tau[44,2] = 0
 tau_computation_44_3: - betas[3] - 0.4730729541736584 gammas[0]
   - 0.9020480301699687 gammas[1] - 0.9315130317672309 gammas[2]
   - 1.390754901918511 gammas[3] + tau[44,3] = 0
 tau_computation_44_4: - betas[4] - 0.4730729541736584 gammas[0]
   - 0.9020480301699687 gammas[1] - 0.9315130317672309 gammas[2]
   - 1.390754901918511 gammas[3] + tau[44,4] = 0
 tau_computation_44_5: - betas[5] - 0.4730729541736584 gammas[0]
   - 0.9020480301699687 gammas[1] - 0.9315130317672309 gammas[2]
   - 1.390754901918511 gammas[3] + tau[44,5] = 0
 tau_computation_44_6: - betas[6] - 0.4730729541736584 gammas[0]
   - 0.9020480301699687 gammas[1] - 0.9315130317672309 gammas[2]
   - 1.390754901918511 gammas[3] + tau[44,6] = 0
 tau_computation_44_7: - betas[7] - 0.4730729541736584 gammas[0]
   - 0.9020480301699687 gammas[1] - 0.9315130317672309 gammas[2]
   - 1.390754901918511 gammas[3] + tau[44,7] = 0
 Ensure_only_one_R_true_44: R[44,0] + R[44,1] + R[44,2] + R[44,3] + R[44,4]
   + R[44,5] + R[44,6] + R[44,7] = 1
 Define_Q_hat_44: - 2198.078009343337 R[44,0] - 2381.076769741246 R[44,1]
   - 2389.430368794413 R[44,2] - 2386.725109272777 R[44,3]
   - 2412.115156824478 R[44,4] - 2412.115166840646 R[44,5]
   - 2281.731853480934 R[44,6] - 2354.348623362157 R[44,7] + Q_hat[44] = 0
 Adjust_Q_hat_44: Q0_var[44] - Q_hat[44] + Q_hat_adjusted[44] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_44: Total_Demand_Up_to_K_minus_1[44]
   - 254.2216927147899 R[44,0] - 483.4366141753967 R[44,1]
   - 761.9569489042894 R[44,2] - 991.1718693895793 R[44,3]
   - 1225.49806389718 R[44,4] - 1454.712984195329 R[44,5]
   - 1580.988109839855 R[44,6] - 1895.918504047045 R[44,7] = 0
 Constr_Total_Demand_from_K_to_T_44: Total_Demand_from_k_to_T[44]
   - 2100.126651928552 R[44,0] - 1870.911730467946 R[44,1]
   - 1592.391395739053 R[44,2] - 1363.176475253763 R[44,3]
   - 1128.850280746162 R[44,4] - 899.6353604480133 R[44,5]
   - 773.3602348034865 R[44,6] - 458.4298405962969 R[44,7] = 0
 Constr_Left_0_diff_aux_44: Sold_0[44] - Q0_var[44] + Left_0_aux_44 = 0
 Constr_Lost_0_diff_aux_44: - Total_Demand_Up_to_K_minus_1[44] + Q0_var[44]
   + Lost_0_aux_44 = 0
 Constr_Q1_plus_left_44: - Left_0[44] + Q1_plus_left[44] - Q1_var[44] = 0
 Constr_Left_1_diff_aux_44: Sold_1[44] - Q1_plus_left[44] + Left_0_aux_44
   = 0
 Constr_Lost_1_diff_aux_44: - Total_Demand_from_k_to_T[44]
   + Q1_plus_left[44] + Lost_0_aux_44 = 0
 Profit_Constraint_44: - 700 Sold_0[44] + 700 Lost_0[44] - 700 Sold_1[44]
   + 300 Left_1[44] + 700 Lost_1[44] + profits_vars[44] = 0
 R945: - 0.4872912432374461 alphas[0] - 0.4843558456788744 alphas[1]
   + 0.2339638849644033 alphas[2] + 1.232863003333669 alphas[3] - alphas[4]
   + f_var[45] = 0
 Q0_upper_bound_45:
   - 2028.416054007841 Fraction_for_second_order_amount[45] + Q0_var[45]
   = 0
 tau_computation_45_0: - betas[0] - 0.4872912432374461 gammas[0]
   - 0.4843558456788744 gammas[1] + 0.2339638849644033 gammas[2]
   + 1.232863003333669 gammas[3] + tau[45,0] = 0
 tau_computation_45_1: - betas[1] - 0.4872912432374461 gammas[0]
   - 0.4843558456788744 gammas[1] + 0.2339638849644033 gammas[2]
   + 1.232863003333669 gammas[3] + tau[45,1] = 0
 tau_computation_45_2: - betas[2] - 0.4872912432374461 gammas[0]
   - 0.4843558456788744 gammas[1] + 0.2339638849644033 gammas[2]
   + 1.232863003333669 gammas[3] + tau[45,2] = 0
 tau_computation_45_3: - betas[3] - 0.4872912432374461 gammas[0]
   - 0.4843558456788744 gammas[1] + 0.2339638849644033 gammas[2]
   + 1.232863003333669 gammas[3] + tau[45,3] = 0
 tau_computation_45_4: - betas[4] - 0.4872912432374461 gammas[0]
   - 0.4843558456788744 gammas[1] + 0.2339638849644033 gammas[2]
   + 1.232863003333669 gammas[3] + tau[45,4] = 0
 tau_computation_45_5: - betas[5] - 0.4872912432374461 gammas[0]
   - 0.4843558456788744 gammas[1] + 0.2339638849644033 gammas[2]
   + 1.232863003333669 gammas[3] + tau[45,5] = 0
 tau_computation_45_6: - betas[6] - 0.4872912432374461 gammas[0]
   - 0.4843558456788744 gammas[1] + 0.2339638849644033 gammas[2]
   + 1.232863003333669 gammas[3] + tau[45,6] = 0
 tau_computation_45_7: - betas[7] - 0.4872912432374461 gammas[0]
   - 0.4843558456788744 gammas[1] + 0.2339638849644033 gammas[2]
   + 1.232863003333669 gammas[3] + tau[45,7] = 0
 Ensure_only_one_R_true_45: R[45,0] + R[45,1] + R[45,2] + R[45,3] + R[45,4]
   + R[45,5] + R[45,6] + R[45,7] = 1
 Define_Q_hat_45: - 2141.183633878529 R[45,0] - 1640.048410656392 R[45,1]
   - 1631.826377197627 R[45,2] - 1626.182549862724 R[45,3]
   - 1547.870699903405 R[45,4] - 1547.87069290753 R[45,5]
   - 1463.180923707809 R[45,6] - 1350.166274010808 R[45,7] + Q_hat[45] = 0
 Adjust_Q_hat_45: Q0_var[45] - Q_hat[45] + Q_hat_adjusted[45] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_45: Total_Demand_Up_to_K_minus_1[45]
   - 235.8650164887246 R[45,0] - 384.8778262547837 R[45,1]
   - 574.6178631118536 R[45,2] - 723.6327887629554 R[45,3]
   - 787.03293343333 R[45,4] - 936.0478591351534 R[45,5]
   - 1026.979500407719 R[45,6] - 1052.136144779188 R[45,7] = 0
 Constr_Total_Demand_from_K_to_T_45: Total_Demand_from_k_to_T[45]
   - 1114.300979694111 R[45,0] - 965.2881699280514 R[45,1]
   - 775.5481330709815 R[45,2] - 626.5332074198795 R[45,3]
   - 563.133062749505 R[45,4] - 414.1181370476816 R[45,5]
   - 323.1864957751158 R[45,6] - 298.0298514036469 R[45,7] = 0
 Constr_Left_0_diff_aux_45: Sold_0[45] - Q0_var[45] + Left_0_aux_45 = 0
 Constr_Lost_0_diff_aux_45: - Total_Demand_Up_to_K_minus_1[45] + Q0_var[45]
   + Lost_0_aux_45 = 0
 Constr_Q1_plus_left_45: - Left_0[45] + Q1_plus_left[45] - Q1_var[45] = 0
 Constr_Left_1_diff_aux_45: Sold_1[45] - Q1_plus_left[45] + Left_0_aux_45
   = 0
 Constr_Lost_1_diff_aux_45: - Total_Demand_from_k_to_T[45]
   + Q1_plus_left[45] + Lost_0_aux_45 = 0
 Profit_Constraint_45: - 700 Sold_0[45] + 700 Lost_0[45] - 700 Sold_1[45]
   + 300 Left_1[45] + 700 Lost_1[45] + profits_vars[45] = 0
 R966: 1.20317178882866 alphas[0] + 1.980272209355848 alphas[1]
   - 0.3813324343285022 alphas[2] - 0.3709798611131848 alphas[3]
   - alphas[4] + f_var[46] = 0
 Q0_upper_bound_46:
   - 2028.416054007841 Fraction_for_second_order_amount[46] + Q0_var[46]
   = 0
 tau_computation_46_0: - betas[0] + 1.20317178882866 gammas[0]
   + 1.980272209355848 gammas[1] - 0.3813324343285022 gammas[2]
   - 0.3709798611131848 gammas[3] + tau[46,0] = 0
 tau_computation_46_1: - betas[1] + 1.20317178882866 gammas[0]
   + 1.980272209355848 gammas[1] - 0.3813324343285022 gammas[2]
   - 0.3709798611131848 gammas[3] + tau[46,1] = 0
 tau_computation_46_2: - betas[2] + 1.20317178882866 gammas[0]
   + 1.980272209355848 gammas[1] - 0.3813324343285022 gammas[2]
   - 0.3709798611131848 gammas[3] + tau[46,2] = 0
 tau_computation_46_3: - betas[3] + 1.20317178882866 gammas[0]
   + 1.980272209355848 gammas[1] - 0.3813324343285022 gammas[2]
   - 0.3709798611131848 gammas[3] + tau[46,3] = 0
 tau_computation_46_4: - betas[4] + 1.20317178882866 gammas[0]
   + 1.980272209355848 gammas[1] - 0.3813324343285022 gammas[2]
   - 0.3709798611131848 gammas[3] + tau[46,4] = 0
 tau_computation_46_5: - betas[5] + 1.20317178882866 gammas[0]
   + 1.980272209355848 gammas[1] - 0.3813324343285022 gammas[2]
   - 0.3709798611131848 gammas[3] + tau[46,5] = 0
 tau_computation_46_6: - betas[6] + 1.20317178882866 gammas[0]
   + 1.980272209355848 gammas[1] - 0.3813324343285022 gammas[2]
   - 0.3709798611131848 gammas[3] + tau[46,6] = 0
 tau_computation_46_7: - betas[7] + 1.20317178882866 gammas[0]
   + 1.980272209355848 gammas[1] - 0.3813324343285022 gammas[2]
   - 0.3709798611131848 gammas[3] + tau[46,7] = 0
 Ensure_only_one_R_true_46: R[46,0] + R[46,1] + R[46,2] + R[46,3] + R[46,4]
   + R[46,5] + R[46,6] + R[46,7] = 1
 Define_Q_hat_46: - 2080.740776931751 R[46,0] - 2015.096230867005 R[46,1]
   - 2051.941821685267 R[46,2] - 2041.102051529088 R[46,3]
   - 2080.603912448339 R[46,4] - 2080.603829095304 R[46,5]
   - 2058.07653565388 R[46,6] - 2058.768596505982 R[46,7] + Q_hat[46] = 0
 Adjust_Q_hat_46: Q0_var[46] - Q_hat[46] + Q_hat_adjusted[46] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_46: Total_Demand_Up_to_K_minus_1[46]
   - 216.3634410502754 R[46,0] - 407.7187627113049 R[46,1]
   - 705.8023778106747 R[46,2] - 897.1577051623721 R[46,3]
   - 1090.36131342308 R[46,4] - 1281.716641387707 R[46,5]
   - 1418.504636516513 R[46,6] - 1676.057662553893 R[46,7] = 0
 Constr_Total_Demand_from_K_to_T_46: Total_Demand_from_k_to_T[46]
   - 1842.404877432873 R[46,0] - 1651.049555771843 R[46,1]
   - 1352.965940672473 R[46,2] - 1161.610613320776 R[46,3]
   - 968.4070050600681 R[46,4] - 777.0516770954408 R[46,5]
   - 640.2636819666345 R[46,6] - 382.7106559292545 R[46,7] = 0
 Constr_Left_0_diff_aux_46: Sold_0[46] - Q0_var[46] + Left_0_aux_46 = 0
 Constr_Lost_0_diff_aux_46: - Total_Demand_Up_to_K_minus_1[46] + Q0_var[46]
   + Lost_0_aux_46 = 0
 Constr_Q1_plus_left_46: - Left_0[46] + Q1_plus_left[46] - Q1_var[46] = 0
 Constr_Left_1_diff_aux_46: Sold_1[46] - Q1_plus_left[46] + Left_0_aux_46
   = 0
 Constr_Lost_1_diff_aux_46: - Total_Demand_from_k_to_T[46]
   + Q1_plus_left[46] + Lost_0_aux_46 = 0
 Profit_Constraint_46: - 700 Sold_0[46] + 700 Lost_0[46] - 700 Sold_1[46]
   + 300 Left_1[46] + 700 Lost_1[46] + profits_vars[46] = 0
 R987: 1.502350960657328 alphas[0] - 1.364582467079067 alphas[1]
   + 1.007000167908843 alphas[2] + 0.3985862233396426 alphas[3] - alphas[4]
   + f_var[47] = 0
 Q0_upper_bound_47:
   - 2028.416054007841 Fraction_for_second_order_amount[47] + Q0_var[47]
   = 0
 tau_computation_47_0: - betas[0] + 1.502350960657328 gammas[0]
   - 1.364582467079067 gammas[1] + 1.007000167908843 gammas[2]
   + 0.3985862233396426 gammas[3] + tau[47,0] = 0
 tau_computation_47_1: - betas[1] + 1.502350960657328 gammas[0]
   - 1.364582467079067 gammas[1] + 1.007000167908843 gammas[2]
   + 0.3985862233396426 gammas[3] + tau[47,1] = 0
 tau_computation_47_2: - betas[2] + 1.502350960657328 gammas[0]
   - 1.364582467079067 gammas[1] + 1.007000167908843 gammas[2]
   + 0.3985862233396426 gammas[3] + tau[47,2] = 0
 tau_computation_47_3: - betas[3] + 1.502350960657328 gammas[0]
   - 1.364582467079067 gammas[1] + 1.007000167908843 gammas[2]
   + 0.3985862233396426 gammas[3] + tau[47,3] = 0
 tau_computation_47_4: - betas[4] + 1.502350960657328 gammas[0]
   - 1.364582467079067 gammas[1] + 1.007000167908843 gammas[2]
   + 0.3985862233396426 gammas[3] + tau[47,4] = 0
 tau_computation_47_5: - betas[5] + 1.502350960657328 gammas[0]
   - 1.364582467079067 gammas[1] + 1.007000167908843 gammas[2]
   + 0.3985862233396426 gammas[3] + tau[47,5] = 0
 tau_computation_47_6: - betas[6] + 1.502350960657328 gammas[0]
   - 1.364582467079067 gammas[1] + 1.007000167908843 gammas[2]
   + 0.3985862233396426 gammas[3] + tau[47,6] = 0
 tau_computation_47_7: - betas[7] + 1.502350960657328 gammas[0]
   - 1.364582467079067 gammas[1] + 1.007000167908843 gammas[2]
   + 0.3985862233396426 gammas[3] + tau[47,7] = 0
 Ensure_only_one_R_true_47: R[47,0] + R[47,1] + R[47,2] + R[47,3] + R[47,4]
   + R[47,5] + R[47,6] + R[47,7] = 1
 Define_Q_hat_47: - 2249.551355043643 R[47,0] - 1175.251182683844 R[47,1]
   - 1193.009380988354 R[47,2] - 1194.623757760601 R[47,3]
   - 1161.09847494492 R[47,4] - 1161.098319930639 R[47,5]
   - 1059.112948275611 R[47,6] - 1093.822610498864 R[47,7] + Q_hat[47] = 0
 Adjust_Q_hat_47: Q0_var[47] - Q_hat[47] + Q_hat_adjusted[47] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_47: Total_Demand_Up_to_K_minus_1[47]
   - 270.8293017020314 R[47,0] - 366.7105112579748 R[47,1]
   - 589.9727302625092 R[47,2] - 685.7926887655082 R[47,3]
   - 685.7926887655082 R[47,4] - 781.6126478190274 R[47,5]
   - 781.6126478190274 R[47,6] - 902.1824135669464 R[47,7] = 0
 Constr_Total_Demand_from_K_to_T_47: Total_Demand_from_k_to_T[47]
   - 822.9930299664925 R[47,0] - 727.1118204105492 R[47,1]
   - 503.8496014060147 R[47,2] - 408.0296429030158 R[47,3]
   - 408.0296429030158 R[47,4] - 312.2096838494965 R[47,5]
   - 312.2096838494965 R[47,6] - 191.6399181015776 R[47,7] = 0
 Constr_Left_0_diff_aux_47: Sold_0[47] - Q0_var[47] + Left_0_aux_47 = 0
 Constr_Lost_0_diff_aux_47: - Total_Demand_Up_to_K_minus_1[47] + Q0_var[47]
   + Lost_0_aux_47 = 0
 Constr_Q1_plus_left_47: - Left_0[47] + Q1_plus_left[47] - Q1_var[47] = 0
 Constr_Left_1_diff_aux_47: Sold_1[47] - Q1_plus_left[47] + Left_0_aux_47
   = 0
 Constr_Lost_1_diff_aux_47: - Total_Demand_from_k_to_T[47]
   + Q1_plus_left[47] + Lost_0_aux_47 = 0
 Profit_Constraint_47: - 700 Sold_0[47] + 700 Lost_0[47] - 700 Sold_1[47]
   + 300 Left_1[47] + 700 Lost_1[47] + profits_vars[47] = 0
 R1008: 0.8173510907951662 alphas[0] - 0.526623881700362 alphas[1]
   + 1.146625172499899 alphas[2] - 0.5840656824573925 alphas[3] - alphas[4]
   + f_var[48] = 0
 Q0_upper_bound_48:
   - 2028.416054007841 Fraction_for_second_order_amount[48] + Q0_var[48]
   = 0
 tau_computation_48_0: - betas[0] + 0.8173510907951662 gammas[0]
   - 0.526623881700362 gammas[1] + 1.146625172499899 gammas[2]
   - 0.5840656824573925 gammas[3] + tau[48,0] = 0
 tau_computation_48_1: - betas[1] + 0.8173510907951662 gammas[0]
   - 0.526623881700362 gammas[1] + 1.146625172499899 gammas[2]
   - 0.5840656824573925 gammas[3] + tau[48,1] = 0
 tau_computation_48_2: - betas[2] + 0.8173510907951662 gammas[0]
   - 0.526623881700362 gammas[1] + 1.146625172499899 gammas[2]
   - 0.5840656824573925 gammas[3] + tau[48,2] = 0
 tau_computation_48_3: - betas[3] + 0.8173510907951662 gammas[0]
   - 0.526623881700362 gammas[1] + 1.146625172499899 gammas[2]
   - 0.5840656824573925 gammas[3] + tau[48,3] = 0
 tau_computation_48_4: - betas[4] + 0.8173510907951662 gammas[0]
   - 0.526623881700362 gammas[1] + 1.146625172499899 gammas[2]
   - 0.5840656824573925 gammas[3] + tau[48,4] = 0
 tau_computation_48_5: - betas[5] + 0.8173510907951662 gammas[0]
   - 0.526623881700362 gammas[1] + 1.146625172499899 gammas[2]
   - 0.5840656824573925 gammas[3] + tau[48,5] = 0
 tau_computation_48_6: - betas[6] + 0.8173510907951662 gammas[0]
   - 0.526623881700362 gammas[1] + 1.146625172499899 gammas[2]
   - 0.5840656824573925 gammas[3] + tau[48,6] = 0
 tau_computation_48_7: - betas[7] + 0.8173510907951662 gammas[0]
   - 0.526623881700362 gammas[1] + 1.146625172499899 gammas[2]
   - 0.5840656824573925 gammas[3] + tau[48,7] = 0
 Ensure_only_one_R_true_48: R[48,0] + R[48,1] + R[48,2] + R[48,3] + R[48,4]
   + R[48,5] + R[48,6] + R[48,7] = 1
 Define_Q_hat_48: - 1793.087112938792 R[48,0] - 1006.721943532601 R[48,1]
   - 944.4305542600728 R[48,2] - 980.6060231725253 R[48,3]
   - 958.4500828993041 R[48,4] - 958.450003028169 R[48,5]
   - 992.2572259476774 R[48,6] - 979.1215670775243 R[48,7] + Q_hat[48] = 0
 Adjust_Q_hat_48: Q0_var[48] - Q_hat[48] + Q_hat_adjusted[48] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_48: Total_Demand_Up_to_K_minus_1[48]
   - 123.5534728939979 R[48,0] - 209.896207469296 R[48,1]
   - 209.896207469296 R[48,2] - 296.1081476893295 R[48,3]
   - 405.3994736908078 R[48,4] - 491.6114135266531 R[48,5]
   - 717.1665145053593 R[48,6] - 806.6974075638116 R[48,7] = 0
 Constr_Total_Demand_from_K_to_T_48: Total_Demand_from_k_to_T[48]
   - 855.567814343068 R[48,0] - 769.2250797677698 R[48,1]
   - 769.2250797677698 R[48,2] - 683.0131395477363 R[48,3]
   - 573.7218135462581 R[48,4] - 487.5098737104128 R[48,5]
   - 261.9547727317065 R[48,6] - 172.4238796732542 R[48,7] = 0
 Constr_Left_0_diff_aux_48: Sold_0[48] - Q0_var[48] + Left_0_aux_48 = 0
 Constr_Lost_0_diff_aux_48: - Total_Demand_Up_to_K_minus_1[48] + Q0_var[48]
   + Lost_0_aux_48 = 0
 Constr_Q1_plus_left_48: - Left_0[48] + Q1_plus_left[48] - Q1_var[48] = 0
 Constr_Left_1_diff_aux_48: Sold_1[48] - Q1_plus_left[48] + Left_0_aux_48
   = 0
 Constr_Lost_1_diff_aux_48: - Total_Demand_from_k_to_T[48]
   + Q1_plus_left[48] + Lost_0_aux_48 = 0
 Profit_Constraint_48: - 700 Sold_0[48] + 700 Lost_0[48] - 700 Sold_1[48]
   + 300 Left_1[48] + 700 Lost_1[48] + profits_vars[48] = 0
 R1029: 0.6400155019933705 alphas[0] - 1.688611157892237 alphas[1]
   - 0.1002627207618077 alphas[2] - 1.970174987241885 alphas[3] - alphas[4]
   + f_var[49] = 0
 Q0_upper_bound_49:
   - 2028.416054007841 Fraction_for_second_order_amount[49] + Q0_var[49]
   = 0
 tau_computation_49_0: - betas[0] + 0.6400155019933705 gammas[0]
   - 1.688611157892237 gammas[1] - 0.1002627207618077 gammas[2]
   - 1.970174987241885 gammas[3] + tau[49,0] = 0
 tau_computation_49_1: - betas[1] + 0.6400155019933705 gammas[0]
   - 1.688611157892237 gammas[1] - 0.1002627207618077 gammas[2]
   - 1.970174987241885 gammas[3] + tau[49,1] = 0
 tau_computation_49_2: - betas[2] + 0.6400155019933705 gammas[0]
   - 1.688611157892237 gammas[1] - 0.1002627207618077 gammas[2]
   - 1.970174987241885 gammas[3] + tau[49,2] = 0
 tau_computation_49_3: - betas[3] + 0.6400155019933705 gammas[0]
   - 1.688611157892237 gammas[1] - 0.1002627207618077 gammas[2]
   - 1.970174987241885 gammas[3] + tau[49,3] = 0
 tau_computation_49_4: - betas[4] + 0.6400155019933705 gammas[0]
   - 1.688611157892237 gammas[1] - 0.1002627207618077 gammas[2]
   - 1.970174987241885 gammas[3] + tau[49,4] = 0
 tau_computation_49_5: - betas[5] + 0.6400155019933705 gammas[0]
   - 1.688611157892237 gammas[1] - 0.1002627207618077 gammas[2]
   - 1.970174987241885 gammas[3] + tau[49,5] = 0
 tau_computation_49_6: - betas[6] + 0.6400155019933705 gammas[0]
   - 1.688611157892237 gammas[1] - 0.1002627207618077 gammas[2]
   - 1.970174987241885 gammas[3] + tau[49,6] = 0
 tau_computation_49_7: - betas[7] + 0.6400155019933705 gammas[0]
   - 1.688611157892237 gammas[1] - 0.1002627207618077 gammas[2]
   - 1.970174987241885 gammas[3] + tau[49,7] = 0
 Ensure_only_one_R_true_49: R[49,0] + R[49,1] + R[49,2] + R[49,3] + R[49,4]
   + R[49,5] + R[49,6] + R[49,7] = 1
 Define_Q_hat_49: - 2227.003644707747 R[49,0] - 1864.994706347642 R[49,1]
   - 1835.358048281501 R[49,2] - 1835.985947926099 R[49,3]
   - 1838.928535118157 R[49,4] - 1838.928506058931 R[49,5]
   - 1555.333465212909 R[49,6] - 1565.025855568255 R[49,7] + Q_hat[49] = 0
 Adjust_Q_hat_49: Q0_var[49] - Q_hat[49] + Q_hat_adjusted[49] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_49: Total_Demand_Up_to_K_minus_1[49]
   - 263.5543994837769 R[49,0] - 435.5684560428369 R[49,1]
   - 607.0136917744279 R[49,2] - 779.0277617174847 R[49,3]
   - 936.12969143525 R[49,4] - 1108.143761597489 R[49,5]
   - 1108.143761597489 R[49,6] - 1220.997436910324 R[49,7] = 0
 Constr_Total_Demand_from_K_to_T_49: Total_Demand_from_k_to_T[49]
   - 1301.471177751024 R[49,0] - 1129.457121191964 R[49,1]
   - 958.011885460373 R[49,2] - 785.9978155173163 R[49,3]
   - 628.895885799551 R[49,4] - 456.8818156373125 R[49,5]
   - 456.8818156373125 R[49,6] - 344.0281403244772 R[49,7] = 0
 Constr_Left_0_diff_aux_49: Sold_0[49] - Q0_var[49] + Left_0_aux_49 = 0
 Constr_Lost_0_diff_aux_49: - Total_Demand_Up_to_K_minus_1[49] + Q0_var[49]
   + Lost_0_aux_49 = 0
 Constr_Q1_plus_left_49: - Left_0[49] + Q1_plus_left[49] - Q1_var[49] = 0
 Constr_Left_1_diff_aux_49: Sold_1[49] - Q1_plus_left[49] + Left_0_aux_49
   = 0
 Constr_Lost_1_diff_aux_49: - Total_Demand_from_k_to_T[49]
   + Q1_plus_left[49] + Lost_0_aux_49 = 0
 Profit_Constraint_49: - 700 Sold_0[49] + 700 Lost_0[49] - 700 Sold_1[49]
   + 300 Left_1[49] + 700 Lost_1[49] + profits_vars[49] = 0
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
 softmax_25_0: - exp_tau_var_25_0 + [ r[25,0] * exp_tau_var_25_0
   + r[25,0] * exp_tau_var_25_1 + r[25,0] * exp_tau_var_25_2
   + r[25,0] * exp_tau_var_25_3 + r[25,0] * exp_tau_var_25_4
   + r[25,0] * exp_tau_var_25_5 + r[25,0] * exp_tau_var_25_6
   + r[25,0] * exp_tau_var_25_7 ] = 0
 softmax_25_1: - exp_tau_var_25_1 + [ r[25,1] * exp_tau_var_25_0
   + r[25,1] * exp_tau_var_25_1 + r[25,1] * exp_tau_var_25_2
   + r[25,1] * exp_tau_var_25_3 + r[25,1] * exp_tau_var_25_4
   + r[25,1] * exp_tau_var_25_5 + r[25,1] * exp_tau_var_25_6
   + r[25,1] * exp_tau_var_25_7 ] = 0
 softmax_25_2: - exp_tau_var_25_2 + [ r[25,2] * exp_tau_var_25_0
   + r[25,2] * exp_tau_var_25_1 + r[25,2] * exp_tau_var_25_2
   + r[25,2] * exp_tau_var_25_3 + r[25,2] * exp_tau_var_25_4
   + r[25,2] * exp_tau_var_25_5 + r[25,2] * exp_tau_var_25_6
   + r[25,2] * exp_tau_var_25_7 ] = 0
 softmax_25_3: - exp_tau_var_25_3 + [ r[25,3] * exp_tau_var_25_0
   + r[25,3] * exp_tau_var_25_1 + r[25,3] * exp_tau_var_25_2
   + r[25,3] * exp_tau_var_25_3 + r[25,3] * exp_tau_var_25_4
   + r[25,3] * exp_tau_var_25_5 + r[25,3] * exp_tau_var_25_6
   + r[25,3] * exp_tau_var_25_7 ] = 0
 softmax_25_4: - exp_tau_var_25_4 + [ r[25,4] * exp_tau_var_25_0
   + r[25,4] * exp_tau_var_25_1 + r[25,4] * exp_tau_var_25_2
   + r[25,4] * exp_tau_var_25_3 + r[25,4] * exp_tau_var_25_4
   + r[25,4] * exp_tau_var_25_5 + r[25,4] * exp_tau_var_25_6
   + r[25,4] * exp_tau_var_25_7 ] = 0
 softmax_25_5: - exp_tau_var_25_5 + [ r[25,5] * exp_tau_var_25_0
   + r[25,5] * exp_tau_var_25_1 + r[25,5] * exp_tau_var_25_2
   + r[25,5] * exp_tau_var_25_3 + r[25,5] * exp_tau_var_25_4
   + r[25,5] * exp_tau_var_25_5 + r[25,5] * exp_tau_var_25_6
   + r[25,5] * exp_tau_var_25_7 ] = 0
 softmax_25_6: - exp_tau_var_25_6 + [ r[25,6] * exp_tau_var_25_0
   + r[25,6] * exp_tau_var_25_1 + r[25,6] * exp_tau_var_25_2
   + r[25,6] * exp_tau_var_25_3 + r[25,6] * exp_tau_var_25_4
   + r[25,6] * exp_tau_var_25_5 + r[25,6] * exp_tau_var_25_6
   + r[25,6] * exp_tau_var_25_7 ] = 0
 softmax_25_7: - exp_tau_var_25_7 + [ r[25,7] * exp_tau_var_25_0
   + r[25,7] * exp_tau_var_25_1 + r[25,7] * exp_tau_var_25_2
   + r[25,7] * exp_tau_var_25_3 + r[25,7] * exp_tau_var_25_4
   + r[25,7] * exp_tau_var_25_5 + r[25,7] * exp_tau_var_25_6
   + r[25,7] * exp_tau_var_25_7 ] = 0
 softmax_26_0: - exp_tau_var_26_0 + [ r[26,0] * exp_tau_var_26_0
   + r[26,0] * exp_tau_var_26_1 + r[26,0] * exp_tau_var_26_2
   + r[26,0] * exp_tau_var_26_3 + r[26,0] * exp_tau_var_26_4
   + r[26,0] * exp_tau_var_26_5 + r[26,0] * exp_tau_var_26_6
   + r[26,0] * exp_tau_var_26_7 ] = 0
 softmax_26_1: - exp_tau_var_26_1 + [ r[26,1] * exp_tau_var_26_0
   + r[26,1] * exp_tau_var_26_1 + r[26,1] * exp_tau_var_26_2
   + r[26,1] * exp_tau_var_26_3 + r[26,1] * exp_tau_var_26_4
   + r[26,1] * exp_tau_var_26_5 + r[26,1] * exp_tau_var_26_6
   + r[26,1] * exp_tau_var_26_7 ] = 0
 softmax_26_2: - exp_tau_var_26_2 + [ r[26,2] * exp_tau_var_26_0
   + r[26,2] * exp_tau_var_26_1 + r[26,2] * exp_tau_var_26_2
   + r[26,2] * exp_tau_var_26_3 + r[26,2] * exp_tau_var_26_4
   + r[26,2] * exp_tau_var_26_5 + r[26,2] * exp_tau_var_26_6
   + r[26,2] * exp_tau_var_26_7 ] = 0
 softmax_26_3: - exp_tau_var_26_3 + [ r[26,3] * exp_tau_var_26_0
   + r[26,3] * exp_tau_var_26_1 + r[26,3] * exp_tau_var_26_2
   + r[26,3] * exp_tau_var_26_3 + r[26,3] * exp_tau_var_26_4
   + r[26,3] * exp_tau_var_26_5 + r[26,3] * exp_tau_var_26_6
   + r[26,3] * exp_tau_var_26_7 ] = 0
 softmax_26_4: - exp_tau_var_26_4 + [ r[26,4] * exp_tau_var_26_0
   + r[26,4] * exp_tau_var_26_1 + r[26,4] * exp_tau_var_26_2
   + r[26,4] * exp_tau_var_26_3 + r[26,4] * exp_tau_var_26_4
   + r[26,4] * exp_tau_var_26_5 + r[26,4] * exp_tau_var_26_6
   + r[26,4] * exp_tau_var_26_7 ] = 0
 softmax_26_5: - exp_tau_var_26_5 + [ r[26,5] * exp_tau_var_26_0
   + r[26,5] * exp_tau_var_26_1 + r[26,5] * exp_tau_var_26_2
   + r[26,5] * exp_tau_var_26_3 + r[26,5] * exp_tau_var_26_4
   + r[26,5] * exp_tau_var_26_5 + r[26,5] * exp_tau_var_26_6
   + r[26,5] * exp_tau_var_26_7 ] = 0
 softmax_26_6: - exp_tau_var_26_6 + [ r[26,6] * exp_tau_var_26_0
   + r[26,6] * exp_tau_var_26_1 + r[26,6] * exp_tau_var_26_2
   + r[26,6] * exp_tau_var_26_3 + r[26,6] * exp_tau_var_26_4
   + r[26,6] * exp_tau_var_26_5 + r[26,6] * exp_tau_var_26_6
   + r[26,6] * exp_tau_var_26_7 ] = 0
 softmax_26_7: - exp_tau_var_26_7 + [ r[26,7] * exp_tau_var_26_0
   + r[26,7] * exp_tau_var_26_1 + r[26,7] * exp_tau_var_26_2
   + r[26,7] * exp_tau_var_26_3 + r[26,7] * exp_tau_var_26_4
   + r[26,7] * exp_tau_var_26_5 + r[26,7] * exp_tau_var_26_6
   + r[26,7] * exp_tau_var_26_7 ] = 0
 softmax_27_0: - exp_tau_var_27_0 + [ r[27,0] * exp_tau_var_27_0
   + r[27,0] * exp_tau_var_27_1 + r[27,0] * exp_tau_var_27_2
   + r[27,0] * exp_tau_var_27_3 + r[27,0] * exp_tau_var_27_4
   + r[27,0] * exp_tau_var_27_5 + r[27,0] * exp_tau_var_27_6
   + r[27,0] * exp_tau_var_27_7 ] = 0
 softmax_27_1: - exp_tau_var_27_1 + [ r[27,1] * exp_tau_var_27_0
   + r[27,1] * exp_tau_var_27_1 + r[27,1] * exp_tau_var_27_2
   + r[27,1] * exp_tau_var_27_3 + r[27,1] * exp_tau_var_27_4
   + r[27,1] * exp_tau_var_27_5 + r[27,1] * exp_tau_var_27_6
   + r[27,1] * exp_tau_var_27_7 ] = 0
 softmax_27_2: - exp_tau_var_27_2 + [ r[27,2] * exp_tau_var_27_0
   + r[27,2] * exp_tau_var_27_1 + r[27,2] * exp_tau_var_27_2
   + r[27,2] * exp_tau_var_27_3 + r[27,2] * exp_tau_var_27_4
   + r[27,2] * exp_tau_var_27_5 + r[27,2] * exp_tau_var_27_6
   + r[27,2] * exp_tau_var_27_7 ] = 0
 softmax_27_3: - exp_tau_var_27_3 + [ r[27,3] * exp_tau_var_27_0
   + r[27,3] * exp_tau_var_27_1 + r[27,3] * exp_tau_var_27_2
   + r[27,3] * exp_tau_var_27_3 + r[27,3] * exp_tau_var_27_4
   + r[27,3] * exp_tau_var_27_5 + r[27,3] * exp_tau_var_27_6
   + r[27,3] * exp_tau_var_27_7 ] = 0
 softmax_27_4: - exp_tau_var_27_4 + [ r[27,4] * exp_tau_var_27_0
   + r[27,4] * exp_tau_var_27_1 + r[27,4] * exp_tau_var_27_2
   + r[27,4] * exp_tau_var_27_3 + r[27,4] * exp_tau_var_27_4
   + r[27,4] * exp_tau_var_27_5 + r[27,4] * exp_tau_var_27_6
   + r[27,4] * exp_tau_var_27_7 ] = 0
 softmax_27_5: - exp_tau_var_27_5 + [ r[27,5] * exp_tau_var_27_0
   + r[27,5] * exp_tau_var_27_1 + r[27,5] * exp_tau_var_27_2
   + r[27,5] * exp_tau_var_27_3 + r[27,5] * exp_tau_var_27_4
   + r[27,5] * exp_tau_var_27_5 + r[27,5] * exp_tau_var_27_6
   + r[27,5] * exp_tau_var_27_7 ] = 0
 softmax_27_6: - exp_tau_var_27_6 + [ r[27,6] * exp_tau_var_27_0
   + r[27,6] * exp_tau_var_27_1 + r[27,6] * exp_tau_var_27_2
   + r[27,6] * exp_tau_var_27_3 + r[27,6] * exp_tau_var_27_4
   + r[27,6] * exp_tau_var_27_5 + r[27,6] * exp_tau_var_27_6
   + r[27,6] * exp_tau_var_27_7 ] = 0
 softmax_27_7: - exp_tau_var_27_7 + [ r[27,7] * exp_tau_var_27_0
   + r[27,7] * exp_tau_var_27_1 + r[27,7] * exp_tau_var_27_2
   + r[27,7] * exp_tau_var_27_3 + r[27,7] * exp_tau_var_27_4
   + r[27,7] * exp_tau_var_27_5 + r[27,7] * exp_tau_var_27_6
   + r[27,7] * exp_tau_var_27_7 ] = 0
 softmax_28_0: - exp_tau_var_28_0 + [ r[28,0] * exp_tau_var_28_0
   + r[28,0] * exp_tau_var_28_1 + r[28,0] * exp_tau_var_28_2
   + r[28,0] * exp_tau_var_28_3 + r[28,0] * exp_tau_var_28_4
   + r[28,0] * exp_tau_var_28_5 + r[28,0] * exp_tau_var_28_6
   + r[28,0] * exp_tau_var_28_7 ] = 0
 softmax_28_1: - exp_tau_var_28_1 + [ r[28,1] * exp_tau_var_28_0
   + r[28,1] * exp_tau_var_28_1 + r[28,1] * exp_tau_var_28_2
   + r[28,1] * exp_tau_var_28_3 + r[28,1] * exp_tau_var_28_4
   + r[28,1] * exp_tau_var_28_5 + r[28,1] * exp_tau_var_28_6
   + r[28,1] * exp_tau_var_28_7 ] = 0
 softmax_28_2: - exp_tau_var_28_2 + [ r[28,2] * exp_tau_var_28_0
   + r[28,2] * exp_tau_var_28_1 + r[28,2] * exp_tau_var_28_2
   + r[28,2] * exp_tau_var_28_3 + r[28,2] * exp_tau_var_28_4
   + r[28,2] * exp_tau_var_28_5 + r[28,2] * exp_tau_var_28_6
   + r[28,2] * exp_tau_var_28_7 ] = 0
 softmax_28_3: - exp_tau_var_28_3 + [ r[28,3] * exp_tau_var_28_0
   + r[28,3] * exp_tau_var_28_1 + r[28,3] * exp_tau_var_28_2
   + r[28,3] * exp_tau_var_28_3 + r[28,3] * exp_tau_var_28_4
   + r[28,3] * exp_tau_var_28_5 + r[28,3] * exp_tau_var_28_6
   + r[28,3] * exp_tau_var_28_7 ] = 0
 softmax_28_4: - exp_tau_var_28_4 + [ r[28,4] * exp_tau_var_28_0
   + r[28,4] * exp_tau_var_28_1 + r[28,4] * exp_tau_var_28_2
   + r[28,4] * exp_tau_var_28_3 + r[28,4] * exp_tau_var_28_4
   + r[28,4] * exp_tau_var_28_5 + r[28,4] * exp_tau_var_28_6
   + r[28,4] * exp_tau_var_28_7 ] = 0
 softmax_28_5: - exp_tau_var_28_5 + [ r[28,5] * exp_tau_var_28_0
   + r[28,5] * exp_tau_var_28_1 + r[28,5] * exp_tau_var_28_2
   + r[28,5] * exp_tau_var_28_3 + r[28,5] * exp_tau_var_28_4
   + r[28,5] * exp_tau_var_28_5 + r[28,5] * exp_tau_var_28_6
   + r[28,5] * exp_tau_var_28_7 ] = 0
 softmax_28_6: - exp_tau_var_28_6 + [ r[28,6] * exp_tau_var_28_0
   + r[28,6] * exp_tau_var_28_1 + r[28,6] * exp_tau_var_28_2
   + r[28,6] * exp_tau_var_28_3 + r[28,6] * exp_tau_var_28_4
   + r[28,6] * exp_tau_var_28_5 + r[28,6] * exp_tau_var_28_6
   + r[28,6] * exp_tau_var_28_7 ] = 0
 softmax_28_7: - exp_tau_var_28_7 + [ r[28,7] * exp_tau_var_28_0
   + r[28,7] * exp_tau_var_28_1 + r[28,7] * exp_tau_var_28_2
   + r[28,7] * exp_tau_var_28_3 + r[28,7] * exp_tau_var_28_4
   + r[28,7] * exp_tau_var_28_5 + r[28,7] * exp_tau_var_28_6
   + r[28,7] * exp_tau_var_28_7 ] = 0
 softmax_29_0: - exp_tau_var_29_0 + [ r[29,0] * exp_tau_var_29_0
   + r[29,0] * exp_tau_var_29_1 + r[29,0] * exp_tau_var_29_2
   + r[29,0] * exp_tau_var_29_3 + r[29,0] * exp_tau_var_29_4
   + r[29,0] * exp_tau_var_29_5 + r[29,0] * exp_tau_var_29_6
   + r[29,0] * exp_tau_var_29_7 ] = 0
 softmax_29_1: - exp_tau_var_29_1 + [ r[29,1] * exp_tau_var_29_0
   + r[29,1] * exp_tau_var_29_1 + r[29,1] * exp_tau_var_29_2
   + r[29,1] * exp_tau_var_29_3 + r[29,1] * exp_tau_var_29_4
   + r[29,1] * exp_tau_var_29_5 + r[29,1] * exp_tau_var_29_6
   + r[29,1] * exp_tau_var_29_7 ] = 0
 softmax_29_2: - exp_tau_var_29_2 + [ r[29,2] * exp_tau_var_29_0
   + r[29,2] * exp_tau_var_29_1 + r[29,2] * exp_tau_var_29_2
   + r[29,2] * exp_tau_var_29_3 + r[29,2] * exp_tau_var_29_4
   + r[29,2] * exp_tau_var_29_5 + r[29,2] * exp_tau_var_29_6
   + r[29,2] * exp_tau_var_29_7 ] = 0
 softmax_29_3: - exp_tau_var_29_3 + [ r[29,3] * exp_tau_var_29_0
   + r[29,3] * exp_tau_var_29_1 + r[29,3] * exp_tau_var_29_2
   + r[29,3] * exp_tau_var_29_3 + r[29,3] * exp_tau_var_29_4
   + r[29,3] * exp_tau_var_29_5 + r[29,3] * exp_tau_var_29_6
   + r[29,3] * exp_tau_var_29_7 ] = 0
 softmax_29_4: - exp_tau_var_29_4 + [ r[29,4] * exp_tau_var_29_0
   + r[29,4] * exp_tau_var_29_1 + r[29,4] * exp_tau_var_29_2
   + r[29,4] * exp_tau_var_29_3 + r[29,4] * exp_tau_var_29_4
   + r[29,4] * exp_tau_var_29_5 + r[29,4] * exp_tau_var_29_6
   + r[29,4] * exp_tau_var_29_7 ] = 0
 softmax_29_5: - exp_tau_var_29_5 + [ r[29,5] * exp_tau_var_29_0
   + r[29,5] * exp_tau_var_29_1 + r[29,5] * exp_tau_var_29_2
   + r[29,5] * exp_tau_var_29_3 + r[29,5] * exp_tau_var_29_4
   + r[29,5] * exp_tau_var_29_5 + r[29,5] * exp_tau_var_29_6
   + r[29,5] * exp_tau_var_29_7 ] = 0
 softmax_29_6: - exp_tau_var_29_6 + [ r[29,6] * exp_tau_var_29_0
   + r[29,6] * exp_tau_var_29_1 + r[29,6] * exp_tau_var_29_2
   + r[29,6] * exp_tau_var_29_3 + r[29,6] * exp_tau_var_29_4
   + r[29,6] * exp_tau_var_29_5 + r[29,6] * exp_tau_var_29_6
   + r[29,6] * exp_tau_var_29_7 ] = 0
 softmax_29_7: - exp_tau_var_29_7 + [ r[29,7] * exp_tau_var_29_0
   + r[29,7] * exp_tau_var_29_1 + r[29,7] * exp_tau_var_29_2
   + r[29,7] * exp_tau_var_29_3 + r[29,7] * exp_tau_var_29_4
   + r[29,7] * exp_tau_var_29_5 + r[29,7] * exp_tau_var_29_6
   + r[29,7] * exp_tau_var_29_7 ] = 0
 softmax_30_0: - exp_tau_var_30_0 + [ r[30,0] * exp_tau_var_30_0
   + r[30,0] * exp_tau_var_30_1 + r[30,0] * exp_tau_var_30_2
   + r[30,0] * exp_tau_var_30_3 + r[30,0] * exp_tau_var_30_4
   + r[30,0] * exp_tau_var_30_5 + r[30,0] * exp_tau_var_30_6
   + r[30,0] * exp_tau_var_30_7 ] = 0
 softmax_30_1: - exp_tau_var_30_1 + [ r[30,1] * exp_tau_var_30_0
   + r[30,1] * exp_tau_var_30_1 + r[30,1] * exp_tau_var_30_2
   + r[30,1] * exp_tau_var_30_3 + r[30,1] * exp_tau_var_30_4
   + r[30,1] * exp_tau_var_30_5 + r[30,1] * exp_tau_var_30_6
   + r[30,1] * exp_tau_var_30_7 ] = 0
 softmax_30_2: - exp_tau_var_30_2 + [ r[30,2] * exp_tau_var_30_0
   + r[30,2] * exp_tau_var_30_1 + r[30,2] * exp_tau_var_30_2
   + r[30,2] * exp_tau_var_30_3 + r[30,2] * exp_tau_var_30_4
   + r[30,2] * exp_tau_var_30_5 + r[30,2] * exp_tau_var_30_6
   + r[30,2] * exp_tau_var_30_7 ] = 0
 softmax_30_3: - exp_tau_var_30_3 + [ r[30,3] * exp_tau_var_30_0
   + r[30,3] * exp_tau_var_30_1 + r[30,3] * exp_tau_var_30_2
   + r[30,3] * exp_tau_var_30_3 + r[30,3] * exp_tau_var_30_4
   + r[30,3] * exp_tau_var_30_5 + r[30,3] * exp_tau_var_30_6
   + r[30,3] * exp_tau_var_30_7 ] = 0
 softmax_30_4: - exp_tau_var_30_4 + [ r[30,4] * exp_tau_var_30_0
   + r[30,4] * exp_tau_var_30_1 + r[30,4] * exp_tau_var_30_2
   + r[30,4] * exp_tau_var_30_3 + r[30,4] * exp_tau_var_30_4
   + r[30,4] * exp_tau_var_30_5 + r[30,4] * exp_tau_var_30_6
   + r[30,4] * exp_tau_var_30_7 ] = 0
 softmax_30_5: - exp_tau_var_30_5 + [ r[30,5] * exp_tau_var_30_0
   + r[30,5] * exp_tau_var_30_1 + r[30,5] * exp_tau_var_30_2
   + r[30,5] * exp_tau_var_30_3 + r[30,5] * exp_tau_var_30_4
   + r[30,5] * exp_tau_var_30_5 + r[30,5] * exp_tau_var_30_6
   + r[30,5] * exp_tau_var_30_7 ] = 0
 softmax_30_6: - exp_tau_var_30_6 + [ r[30,6] * exp_tau_var_30_0
   + r[30,6] * exp_tau_var_30_1 + r[30,6] * exp_tau_var_30_2
   + r[30,6] * exp_tau_var_30_3 + r[30,6] * exp_tau_var_30_4
   + r[30,6] * exp_tau_var_30_5 + r[30,6] * exp_tau_var_30_6
   + r[30,6] * exp_tau_var_30_7 ] = 0
 softmax_30_7: - exp_tau_var_30_7 + [ r[30,7] * exp_tau_var_30_0
   + r[30,7] * exp_tau_var_30_1 + r[30,7] * exp_tau_var_30_2
   + r[30,7] * exp_tau_var_30_3 + r[30,7] * exp_tau_var_30_4
   + r[30,7] * exp_tau_var_30_5 + r[30,7] * exp_tau_var_30_6
   + r[30,7] * exp_tau_var_30_7 ] = 0
 softmax_31_0: - exp_tau_var_31_0 + [ r[31,0] * exp_tau_var_31_0
   + r[31,0] * exp_tau_var_31_1 + r[31,0] * exp_tau_var_31_2
   + r[31,0] * exp_tau_var_31_3 + r[31,0] * exp_tau_var_31_4
   + r[31,0] * exp_tau_var_31_5 + r[31,0] * exp_tau_var_31_6
   + r[31,0] * exp_tau_var_31_7 ] = 0
 softmax_31_1: - exp_tau_var_31_1 + [ r[31,1] * exp_tau_var_31_0
   + r[31,1] * exp_tau_var_31_1 + r[31,1] * exp_tau_var_31_2
   + r[31,1] * exp_tau_var_31_3 + r[31,1] * exp_tau_var_31_4
   + r[31,1] * exp_tau_var_31_5 + r[31,1] * exp_tau_var_31_6
   + r[31,1] * exp_tau_var_31_7 ] = 0
 softmax_31_2: - exp_tau_var_31_2 + [ r[31,2] * exp_tau_var_31_0
   + r[31,2] * exp_tau_var_31_1 + r[31,2] * exp_tau_var_31_2
   + r[31,2] * exp_tau_var_31_3 + r[31,2] * exp_tau_var_31_4
   + r[31,2] * exp_tau_var_31_5 + r[31,2] * exp_tau_var_31_6
   + r[31,2] * exp_tau_var_31_7 ] = 0
 softmax_31_3: - exp_tau_var_31_3 + [ r[31,3] * exp_tau_var_31_0
   + r[31,3] * exp_tau_var_31_1 + r[31,3] * exp_tau_var_31_2
   + r[31,3] * exp_tau_var_31_3 + r[31,3] * exp_tau_var_31_4
   + r[31,3] * exp_tau_var_31_5 + r[31,3] * exp_tau_var_31_6
   + r[31,3] * exp_tau_var_31_7 ] = 0
 softmax_31_4: - exp_tau_var_31_4 + [ r[31,4] * exp_tau_var_31_0
   + r[31,4] * exp_tau_var_31_1 + r[31,4] * exp_tau_var_31_2
   + r[31,4] * exp_tau_var_31_3 + r[31,4] * exp_tau_var_31_4
   + r[31,4] * exp_tau_var_31_5 + r[31,4] * exp_tau_var_31_6
   + r[31,4] * exp_tau_var_31_7 ] = 0
 softmax_31_5: - exp_tau_var_31_5 + [ r[31,5] * exp_tau_var_31_0
   + r[31,5] * exp_tau_var_31_1 + r[31,5] * exp_tau_var_31_2
   + r[31,5] * exp_tau_var_31_3 + r[31,5] * exp_tau_var_31_4
   + r[31,5] * exp_tau_var_31_5 + r[31,5] * exp_tau_var_31_6
   + r[31,5] * exp_tau_var_31_7 ] = 0
 softmax_31_6: - exp_tau_var_31_6 + [ r[31,6] * exp_tau_var_31_0
   + r[31,6] * exp_tau_var_31_1 + r[31,6] * exp_tau_var_31_2
   + r[31,6] * exp_tau_var_31_3 + r[31,6] * exp_tau_var_31_4
   + r[31,6] * exp_tau_var_31_5 + r[31,6] * exp_tau_var_31_6
   + r[31,6] * exp_tau_var_31_7 ] = 0
 softmax_31_7: - exp_tau_var_31_7 + [ r[31,7] * exp_tau_var_31_0
   + r[31,7] * exp_tau_var_31_1 + r[31,7] * exp_tau_var_31_2
   + r[31,7] * exp_tau_var_31_3 + r[31,7] * exp_tau_var_31_4
   + r[31,7] * exp_tau_var_31_5 + r[31,7] * exp_tau_var_31_6
   + r[31,7] * exp_tau_var_31_7 ] = 0
 softmax_32_0: - exp_tau_var_32_0 + [ r[32,0] * exp_tau_var_32_0
   + r[32,0] * exp_tau_var_32_1 + r[32,0] * exp_tau_var_32_2
   + r[32,0] * exp_tau_var_32_3 + r[32,0] * exp_tau_var_32_4
   + r[32,0] * exp_tau_var_32_5 + r[32,0] * exp_tau_var_32_6
   + r[32,0] * exp_tau_var_32_7 ] = 0
 softmax_32_1: - exp_tau_var_32_1 + [ r[32,1] * exp_tau_var_32_0
   + r[32,1] * exp_tau_var_32_1 + r[32,1] * exp_tau_var_32_2
   + r[32,1] * exp_tau_var_32_3 + r[32,1] * exp_tau_var_32_4
   + r[32,1] * exp_tau_var_32_5 + r[32,1] * exp_tau_var_32_6
   + r[32,1] * exp_tau_var_32_7 ] = 0
 softmax_32_2: - exp_tau_var_32_2 + [ r[32,2] * exp_tau_var_32_0
   + r[32,2] * exp_tau_var_32_1 + r[32,2] * exp_tau_var_32_2
   + r[32,2] * exp_tau_var_32_3 + r[32,2] * exp_tau_var_32_4
   + r[32,2] * exp_tau_var_32_5 + r[32,2] * exp_tau_var_32_6
   + r[32,2] * exp_tau_var_32_7 ] = 0
 softmax_32_3: - exp_tau_var_32_3 + [ r[32,3] * exp_tau_var_32_0
   + r[32,3] * exp_tau_var_32_1 + r[32,3] * exp_tau_var_32_2
   + r[32,3] * exp_tau_var_32_3 + r[32,3] * exp_tau_var_32_4
   + r[32,3] * exp_tau_var_32_5 + r[32,3] * exp_tau_var_32_6
   + r[32,3] * exp_tau_var_32_7 ] = 0
 softmax_32_4: - exp_tau_var_32_4 + [ r[32,4] * exp_tau_var_32_0
   + r[32,4] * exp_tau_var_32_1 + r[32,4] * exp_tau_var_32_2
   + r[32,4] * exp_tau_var_32_3 + r[32,4] * exp_tau_var_32_4
   + r[32,4] * exp_tau_var_32_5 + r[32,4] * exp_tau_var_32_6
   + r[32,4] * exp_tau_var_32_7 ] = 0
 softmax_32_5: - exp_tau_var_32_5 + [ r[32,5] * exp_tau_var_32_0
   + r[32,5] * exp_tau_var_32_1 + r[32,5] * exp_tau_var_32_2
   + r[32,5] * exp_tau_var_32_3 + r[32,5] * exp_tau_var_32_4
   + r[32,5] * exp_tau_var_32_5 + r[32,5] * exp_tau_var_32_6
   + r[32,5] * exp_tau_var_32_7 ] = 0
 softmax_32_6: - exp_tau_var_32_6 + [ r[32,6] * exp_tau_var_32_0
   + r[32,6] * exp_tau_var_32_1 + r[32,6] * exp_tau_var_32_2
   + r[32,6] * exp_tau_var_32_3 + r[32,6] * exp_tau_var_32_4
   + r[32,6] * exp_tau_var_32_5 + r[32,6] * exp_tau_var_32_6
   + r[32,6] * exp_tau_var_32_7 ] = 0
 softmax_32_7: - exp_tau_var_32_7 + [ r[32,7] * exp_tau_var_32_0
   + r[32,7] * exp_tau_var_32_1 + r[32,7] * exp_tau_var_32_2
   + r[32,7] * exp_tau_var_32_3 + r[32,7] * exp_tau_var_32_4
   + r[32,7] * exp_tau_var_32_5 + r[32,7] * exp_tau_var_32_6
   + r[32,7] * exp_tau_var_32_7 ] = 0
 softmax_33_0: - exp_tau_var_33_0 + [ r[33,0] * exp_tau_var_33_0
   + r[33,0] * exp_tau_var_33_1 + r[33,0] * exp_tau_var_33_2
   + r[33,0] * exp_tau_var_33_3 + r[33,0] * exp_tau_var_33_4
   + r[33,0] * exp_tau_var_33_5 + r[33,0] * exp_tau_var_33_6
   + r[33,0] * exp_tau_var_33_7 ] = 0
 softmax_33_1: - exp_tau_var_33_1 + [ r[33,1] * exp_tau_var_33_0
   + r[33,1] * exp_tau_var_33_1 + r[33,1] * exp_tau_var_33_2
   + r[33,1] * exp_tau_var_33_3 + r[33,1] * exp_tau_var_33_4
   + r[33,1] * exp_tau_var_33_5 + r[33,1] * exp_tau_var_33_6
   + r[33,1] * exp_tau_var_33_7 ] = 0
 softmax_33_2: - exp_tau_var_33_2 + [ r[33,2] * exp_tau_var_33_0
   + r[33,2] * exp_tau_var_33_1 + r[33,2] * exp_tau_var_33_2
   + r[33,2] * exp_tau_var_33_3 + r[33,2] * exp_tau_var_33_4
   + r[33,2] * exp_tau_var_33_5 + r[33,2] * exp_tau_var_33_6
   + r[33,2] * exp_tau_var_33_7 ] = 0
 softmax_33_3: - exp_tau_var_33_3 + [ r[33,3] * exp_tau_var_33_0
   + r[33,3] * exp_tau_var_33_1 + r[33,3] * exp_tau_var_33_2
   + r[33,3] * exp_tau_var_33_3 + r[33,3] * exp_tau_var_33_4
   + r[33,3] * exp_tau_var_33_5 + r[33,3] * exp_tau_var_33_6
   + r[33,3] * exp_tau_var_33_7 ] = 0
 softmax_33_4: - exp_tau_var_33_4 + [ r[33,4] * exp_tau_var_33_0
   + r[33,4] * exp_tau_var_33_1 + r[33,4] * exp_tau_var_33_2
   + r[33,4] * exp_tau_var_33_3 + r[33,4] * exp_tau_var_33_4
   + r[33,4] * exp_tau_var_33_5 + r[33,4] * exp_tau_var_33_6
   + r[33,4] * exp_tau_var_33_7 ] = 0
 softmax_33_5: - exp_tau_var_33_5 + [ r[33,5] * exp_tau_var_33_0
   + r[33,5] * exp_tau_var_33_1 + r[33,5] * exp_tau_var_33_2
   + r[33,5] * exp_tau_var_33_3 + r[33,5] * exp_tau_var_33_4
   + r[33,5] * exp_tau_var_33_5 + r[33,5] * exp_tau_var_33_6
   + r[33,5] * exp_tau_var_33_7 ] = 0
 softmax_33_6: - exp_tau_var_33_6 + [ r[33,6] * exp_tau_var_33_0
   + r[33,6] * exp_tau_var_33_1 + r[33,6] * exp_tau_var_33_2
   + r[33,6] * exp_tau_var_33_3 + r[33,6] * exp_tau_var_33_4
   + r[33,6] * exp_tau_var_33_5 + r[33,6] * exp_tau_var_33_6
   + r[33,6] * exp_tau_var_33_7 ] = 0
 softmax_33_7: - exp_tau_var_33_7 + [ r[33,7] * exp_tau_var_33_0
   + r[33,7] * exp_tau_var_33_1 + r[33,7] * exp_tau_var_33_2
   + r[33,7] * exp_tau_var_33_3 + r[33,7] * exp_tau_var_33_4
   + r[33,7] * exp_tau_var_33_5 + r[33,7] * exp_tau_var_33_6
   + r[33,7] * exp_tau_var_33_7 ] = 0
 softmax_34_0: - exp_tau_var_34_0 + [ r[34,0] * exp_tau_var_34_0
   + r[34,0] * exp_tau_var_34_1 + r[34,0] * exp_tau_var_34_2
   + r[34,0] * exp_tau_var_34_3 + r[34,0] * exp_tau_var_34_4
   + r[34,0] * exp_tau_var_34_5 + r[34,0] * exp_tau_var_34_6
   + r[34,0] * exp_tau_var_34_7 ] = 0
 softmax_34_1: - exp_tau_var_34_1 + [ r[34,1] * exp_tau_var_34_0
   + r[34,1] * exp_tau_var_34_1 + r[34,1] * exp_tau_var_34_2
   + r[34,1] * exp_tau_var_34_3 + r[34,1] * exp_tau_var_34_4
   + r[34,1] * exp_tau_var_34_5 + r[34,1] * exp_tau_var_34_6
   + r[34,1] * exp_tau_var_34_7 ] = 0
 softmax_34_2: - exp_tau_var_34_2 + [ r[34,2] * exp_tau_var_34_0
   + r[34,2] * exp_tau_var_34_1 + r[34,2] * exp_tau_var_34_2
   + r[34,2] * exp_tau_var_34_3 + r[34,2] * exp_tau_var_34_4
   + r[34,2] * exp_tau_var_34_5 + r[34,2] * exp_tau_var_34_6
   + r[34,2] * exp_tau_var_34_7 ] = 0
 softmax_34_3: - exp_tau_var_34_3 + [ r[34,3] * exp_tau_var_34_0
   + r[34,3] * exp_tau_var_34_1 + r[34,3] * exp_tau_var_34_2
   + r[34,3] * exp_tau_var_34_3 + r[34,3] * exp_tau_var_34_4
   + r[34,3] * exp_tau_var_34_5 + r[34,3] * exp_tau_var_34_6
   + r[34,3] * exp_tau_var_34_7 ] = 0
 softmax_34_4: - exp_tau_var_34_4 + [ r[34,4] * exp_tau_var_34_0
   + r[34,4] * exp_tau_var_34_1 + r[34,4] * exp_tau_var_34_2
   + r[34,4] * exp_tau_var_34_3 + r[34,4] * exp_tau_var_34_4
   + r[34,4] * exp_tau_var_34_5 + r[34,4] * exp_tau_var_34_6
   + r[34,4] * exp_tau_var_34_7 ] = 0
 softmax_34_5: - exp_tau_var_34_5 + [ r[34,5] * exp_tau_var_34_0
   + r[34,5] * exp_tau_var_34_1 + r[34,5] * exp_tau_var_34_2
   + r[34,5] * exp_tau_var_34_3 + r[34,5] * exp_tau_var_34_4
   + r[34,5] * exp_tau_var_34_5 + r[34,5] * exp_tau_var_34_6
   + r[34,5] * exp_tau_var_34_7 ] = 0
 softmax_34_6: - exp_tau_var_34_6 + [ r[34,6] * exp_tau_var_34_0
   + r[34,6] * exp_tau_var_34_1 + r[34,6] * exp_tau_var_34_2
   + r[34,6] * exp_tau_var_34_3 + r[34,6] * exp_tau_var_34_4
   + r[34,6] * exp_tau_var_34_5 + r[34,6] * exp_tau_var_34_6
   + r[34,6] * exp_tau_var_34_7 ] = 0
 softmax_34_7: - exp_tau_var_34_7 + [ r[34,7] * exp_tau_var_34_0
   + r[34,7] * exp_tau_var_34_1 + r[34,7] * exp_tau_var_34_2
   + r[34,7] * exp_tau_var_34_3 + r[34,7] * exp_tau_var_34_4
   + r[34,7] * exp_tau_var_34_5 + r[34,7] * exp_tau_var_34_6
   + r[34,7] * exp_tau_var_34_7 ] = 0
 softmax_35_0: - exp_tau_var_35_0 + [ r[35,0] * exp_tau_var_35_0
   + r[35,0] * exp_tau_var_35_1 + r[35,0] * exp_tau_var_35_2
   + r[35,0] * exp_tau_var_35_3 + r[35,0] * exp_tau_var_35_4
   + r[35,0] * exp_tau_var_35_5 + r[35,0] * exp_tau_var_35_6
   + r[35,0] * exp_tau_var_35_7 ] = 0
 softmax_35_1: - exp_tau_var_35_1 + [ r[35,1] * exp_tau_var_35_0
   + r[35,1] * exp_tau_var_35_1 + r[35,1] * exp_tau_var_35_2
   + r[35,1] * exp_tau_var_35_3 + r[35,1] * exp_tau_var_35_4
   + r[35,1] * exp_tau_var_35_5 + r[35,1] * exp_tau_var_35_6
   + r[35,1] * exp_tau_var_35_7 ] = 0
 softmax_35_2: - exp_tau_var_35_2 + [ r[35,2] * exp_tau_var_35_0
   + r[35,2] * exp_tau_var_35_1 + r[35,2] * exp_tau_var_35_2
   + r[35,2] * exp_tau_var_35_3 + r[35,2] * exp_tau_var_35_4
   + r[35,2] * exp_tau_var_35_5 + r[35,2] * exp_tau_var_35_6
   + r[35,2] * exp_tau_var_35_7 ] = 0
 softmax_35_3: - exp_tau_var_35_3 + [ r[35,3] * exp_tau_var_35_0
   + r[35,3] * exp_tau_var_35_1 + r[35,3] * exp_tau_var_35_2
   + r[35,3] * exp_tau_var_35_3 + r[35,3] * exp_tau_var_35_4
   + r[35,3] * exp_tau_var_35_5 + r[35,3] * exp_tau_var_35_6
   + r[35,3] * exp_tau_var_35_7 ] = 0
 softmax_35_4: - exp_tau_var_35_4 + [ r[35,4] * exp_tau_var_35_0
   + r[35,4] * exp_tau_var_35_1 + r[35,4] * exp_tau_var_35_2
   + r[35,4] * exp_tau_var_35_3 + r[35,4] * exp_tau_var_35_4
   + r[35,4] * exp_tau_var_35_5 + r[35,4] * exp_tau_var_35_6
   + r[35,4] * exp_tau_var_35_7 ] = 0
 softmax_35_5: - exp_tau_var_35_5 + [ r[35,5] * exp_tau_var_35_0
   + r[35,5] * exp_tau_var_35_1 + r[35,5] * exp_tau_var_35_2
   + r[35,5] * exp_tau_var_35_3 + r[35,5] * exp_tau_var_35_4
   + r[35,5] * exp_tau_var_35_5 + r[35,5] * exp_tau_var_35_6
   + r[35,5] * exp_tau_var_35_7 ] = 0
 softmax_35_6: - exp_tau_var_35_6 + [ r[35,6] * exp_tau_var_35_0
   + r[35,6] * exp_tau_var_35_1 + r[35,6] * exp_tau_var_35_2
   + r[35,6] * exp_tau_var_35_3 + r[35,6] * exp_tau_var_35_4
   + r[35,6] * exp_tau_var_35_5 + r[35,6] * exp_tau_var_35_6
   + r[35,6] * exp_tau_var_35_7 ] = 0
 softmax_35_7: - exp_tau_var_35_7 + [ r[35,7] * exp_tau_var_35_0
   + r[35,7] * exp_tau_var_35_1 + r[35,7] * exp_tau_var_35_2
   + r[35,7] * exp_tau_var_35_3 + r[35,7] * exp_tau_var_35_4
   + r[35,7] * exp_tau_var_35_5 + r[35,7] * exp_tau_var_35_6
   + r[35,7] * exp_tau_var_35_7 ] = 0
 softmax_36_0: - exp_tau_var_36_0 + [ r[36,0] * exp_tau_var_36_0
   + r[36,0] * exp_tau_var_36_1 + r[36,0] * exp_tau_var_36_2
   + r[36,0] * exp_tau_var_36_3 + r[36,0] * exp_tau_var_36_4
   + r[36,0] * exp_tau_var_36_5 + r[36,0] * exp_tau_var_36_6
   + r[36,0] * exp_tau_var_36_7 ] = 0
 softmax_36_1: - exp_tau_var_36_1 + [ r[36,1] * exp_tau_var_36_0
   + r[36,1] * exp_tau_var_36_1 + r[36,1] * exp_tau_var_36_2
   + r[36,1] * exp_tau_var_36_3 + r[36,1] * exp_tau_var_36_4
   + r[36,1] * exp_tau_var_36_5 + r[36,1] * exp_tau_var_36_6
   + r[36,1] * exp_tau_var_36_7 ] = 0
 softmax_36_2: - exp_tau_var_36_2 + [ r[36,2] * exp_tau_var_36_0
   + r[36,2] * exp_tau_var_36_1 + r[36,2] * exp_tau_var_36_2
   + r[36,2] * exp_tau_var_36_3 + r[36,2] * exp_tau_var_36_4
   + r[36,2] * exp_tau_var_36_5 + r[36,2] * exp_tau_var_36_6
   + r[36,2] * exp_tau_var_36_7 ] = 0
 softmax_36_3: - exp_tau_var_36_3 + [ r[36,3] * exp_tau_var_36_0
   + r[36,3] * exp_tau_var_36_1 + r[36,3] * exp_tau_var_36_2
   + r[36,3] * exp_tau_var_36_3 + r[36,3] * exp_tau_var_36_4
   + r[36,3] * exp_tau_var_36_5 + r[36,3] * exp_tau_var_36_6
   + r[36,3] * exp_tau_var_36_7 ] = 0
 softmax_36_4: - exp_tau_var_36_4 + [ r[36,4] * exp_tau_var_36_0
   + r[36,4] * exp_tau_var_36_1 + r[36,4] * exp_tau_var_36_2
   + r[36,4] * exp_tau_var_36_3 + r[36,4] * exp_tau_var_36_4
   + r[36,4] * exp_tau_var_36_5 + r[36,4] * exp_tau_var_36_6
   + r[36,4] * exp_tau_var_36_7 ] = 0
 softmax_36_5: - exp_tau_var_36_5 + [ r[36,5] * exp_tau_var_36_0
   + r[36,5] * exp_tau_var_36_1 + r[36,5] * exp_tau_var_36_2
   + r[36,5] * exp_tau_var_36_3 + r[36,5] * exp_tau_var_36_4
   + r[36,5] * exp_tau_var_36_5 + r[36,5] * exp_tau_var_36_6
   + r[36,5] * exp_tau_var_36_7 ] = 0
 softmax_36_6: - exp_tau_var_36_6 + [ r[36,6] * exp_tau_var_36_0
   + r[36,6] * exp_tau_var_36_1 + r[36,6] * exp_tau_var_36_2
   + r[36,6] * exp_tau_var_36_3 + r[36,6] * exp_tau_var_36_4
   + r[36,6] * exp_tau_var_36_5 + r[36,6] * exp_tau_var_36_6
   + r[36,6] * exp_tau_var_36_7 ] = 0
 softmax_36_7: - exp_tau_var_36_7 + [ r[36,7] * exp_tau_var_36_0
   + r[36,7] * exp_tau_var_36_1 + r[36,7] * exp_tau_var_36_2
   + r[36,7] * exp_tau_var_36_3 + r[36,7] * exp_tau_var_36_4
   + r[36,7] * exp_tau_var_36_5 + r[36,7] * exp_tau_var_36_6
   + r[36,7] * exp_tau_var_36_7 ] = 0
 softmax_37_0: - exp_tau_var_37_0 + [ r[37,0] * exp_tau_var_37_0
   + r[37,0] * exp_tau_var_37_1 + r[37,0] * exp_tau_var_37_2
   + r[37,0] * exp_tau_var_37_3 + r[37,0] * exp_tau_var_37_4
   + r[37,0] * exp_tau_var_37_5 + r[37,0] * exp_tau_var_37_6
   + r[37,0] * exp_tau_var_37_7 ] = 0
 softmax_37_1: - exp_tau_var_37_1 + [ r[37,1] * exp_tau_var_37_0
   + r[37,1] * exp_tau_var_37_1 + r[37,1] * exp_tau_var_37_2
   + r[37,1] * exp_tau_var_37_3 + r[37,1] * exp_tau_var_37_4
   + r[37,1] * exp_tau_var_37_5 + r[37,1] * exp_tau_var_37_6
   + r[37,1] * exp_tau_var_37_7 ] = 0
 softmax_37_2: - exp_tau_var_37_2 + [ r[37,2] * exp_tau_var_37_0
   + r[37,2] * exp_tau_var_37_1 + r[37,2] * exp_tau_var_37_2
   + r[37,2] * exp_tau_var_37_3 + r[37,2] * exp_tau_var_37_4
   + r[37,2] * exp_tau_var_37_5 + r[37,2] * exp_tau_var_37_6
   + r[37,2] * exp_tau_var_37_7 ] = 0
 softmax_37_3: - exp_tau_var_37_3 + [ r[37,3] * exp_tau_var_37_0
   + r[37,3] * exp_tau_var_37_1 + r[37,3] * exp_tau_var_37_2
   + r[37,3] * exp_tau_var_37_3 + r[37,3] * exp_tau_var_37_4
   + r[37,3] * exp_tau_var_37_5 + r[37,3] * exp_tau_var_37_6
   + r[37,3] * exp_tau_var_37_7 ] = 0
 softmax_37_4: - exp_tau_var_37_4 + [ r[37,4] * exp_tau_var_37_0
   + r[37,4] * exp_tau_var_37_1 + r[37,4] * exp_tau_var_37_2
   + r[37,4] * exp_tau_var_37_3 + r[37,4] * exp_tau_var_37_4
   + r[37,4] * exp_tau_var_37_5 + r[37,4] * exp_tau_var_37_6
   + r[37,4] * exp_tau_var_37_7 ] = 0
 softmax_37_5: - exp_tau_var_37_5 + [ r[37,5] * exp_tau_var_37_0
   + r[37,5] * exp_tau_var_37_1 + r[37,5] * exp_tau_var_37_2
   + r[37,5] * exp_tau_var_37_3 + r[37,5] * exp_tau_var_37_4
   + r[37,5] * exp_tau_var_37_5 + r[37,5] * exp_tau_var_37_6
   + r[37,5] * exp_tau_var_37_7 ] = 0
 softmax_37_6: - exp_tau_var_37_6 + [ r[37,6] * exp_tau_var_37_0
   + r[37,6] * exp_tau_var_37_1 + r[37,6] * exp_tau_var_37_2
   + r[37,6] * exp_tau_var_37_3 + r[37,6] * exp_tau_var_37_4
   + r[37,6] * exp_tau_var_37_5 + r[37,6] * exp_tau_var_37_6
   + r[37,6] * exp_tau_var_37_7 ] = 0
 softmax_37_7: - exp_tau_var_37_7 + [ r[37,7] * exp_tau_var_37_0
   + r[37,7] * exp_tau_var_37_1 + r[37,7] * exp_tau_var_37_2
   + r[37,7] * exp_tau_var_37_3 + r[37,7] * exp_tau_var_37_4
   + r[37,7] * exp_tau_var_37_5 + r[37,7] * exp_tau_var_37_6
   + r[37,7] * exp_tau_var_37_7 ] = 0
 softmax_38_0: - exp_tau_var_38_0 + [ r[38,0] * exp_tau_var_38_0
   + r[38,0] * exp_tau_var_38_1 + r[38,0] * exp_tau_var_38_2
   + r[38,0] * exp_tau_var_38_3 + r[38,0] * exp_tau_var_38_4
   + r[38,0] * exp_tau_var_38_5 + r[38,0] * exp_tau_var_38_6
   + r[38,0] * exp_tau_var_38_7 ] = 0
 softmax_38_1: - exp_tau_var_38_1 + [ r[38,1] * exp_tau_var_38_0
   + r[38,1] * exp_tau_var_38_1 + r[38,1] * exp_tau_var_38_2
   + r[38,1] * exp_tau_var_38_3 + r[38,1] * exp_tau_var_38_4
   + r[38,1] * exp_tau_var_38_5 + r[38,1] * exp_tau_var_38_6
   + r[38,1] * exp_tau_var_38_7 ] = 0
 softmax_38_2: - exp_tau_var_38_2 + [ r[38,2] * exp_tau_var_38_0
   + r[38,2] * exp_tau_var_38_1 + r[38,2] * exp_tau_var_38_2
   + r[38,2] * exp_tau_var_38_3 + r[38,2] * exp_tau_var_38_4
   + r[38,2] * exp_tau_var_38_5 + r[38,2] * exp_tau_var_38_6
   + r[38,2] * exp_tau_var_38_7 ] = 0
 softmax_38_3: - exp_tau_var_38_3 + [ r[38,3] * exp_tau_var_38_0
   + r[38,3] * exp_tau_var_38_1 + r[38,3] * exp_tau_var_38_2
   + r[38,3] * exp_tau_var_38_3 + r[38,3] * exp_tau_var_38_4
   + r[38,3] * exp_tau_var_38_5 + r[38,3] * exp_tau_var_38_6
   + r[38,3] * exp_tau_var_38_7 ] = 0
 softmax_38_4: - exp_tau_var_38_4 + [ r[38,4] * exp_tau_var_38_0
   + r[38,4] * exp_tau_var_38_1 + r[38,4] * exp_tau_var_38_2
   + r[38,4] * exp_tau_var_38_3 + r[38,4] * exp_tau_var_38_4
   + r[38,4] * exp_tau_var_38_5 + r[38,4] * exp_tau_var_38_6
   + r[38,4] * exp_tau_var_38_7 ] = 0
 softmax_38_5: - exp_tau_var_38_5 + [ r[38,5] * exp_tau_var_38_0
   + r[38,5] * exp_tau_var_38_1 + r[38,5] * exp_tau_var_38_2
   + r[38,5] * exp_tau_var_38_3 + r[38,5] * exp_tau_var_38_4
   + r[38,5] * exp_tau_var_38_5 + r[38,5] * exp_tau_var_38_6
   + r[38,5] * exp_tau_var_38_7 ] = 0
 softmax_38_6: - exp_tau_var_38_6 + [ r[38,6] * exp_tau_var_38_0
   + r[38,6] * exp_tau_var_38_1 + r[38,6] * exp_tau_var_38_2
   + r[38,6] * exp_tau_var_38_3 + r[38,6] * exp_tau_var_38_4
   + r[38,6] * exp_tau_var_38_5 + r[38,6] * exp_tau_var_38_6
   + r[38,6] * exp_tau_var_38_7 ] = 0
 softmax_38_7: - exp_tau_var_38_7 + [ r[38,7] * exp_tau_var_38_0
   + r[38,7] * exp_tau_var_38_1 + r[38,7] * exp_tau_var_38_2
   + r[38,7] * exp_tau_var_38_3 + r[38,7] * exp_tau_var_38_4
   + r[38,7] * exp_tau_var_38_5 + r[38,7] * exp_tau_var_38_6
   + r[38,7] * exp_tau_var_38_7 ] = 0
 softmax_39_0: - exp_tau_var_39_0 + [ r[39,0] * exp_tau_var_39_0
   + r[39,0] * exp_tau_var_39_1 + r[39,0] * exp_tau_var_39_2
   + r[39,0] * exp_tau_var_39_3 + r[39,0] * exp_tau_var_39_4
   + r[39,0] * exp_tau_var_39_5 + r[39,0] * exp_tau_var_39_6
   + r[39,0] * exp_tau_var_39_7 ] = 0
 softmax_39_1: - exp_tau_var_39_1 + [ r[39,1] * exp_tau_var_39_0
   + r[39,1] * exp_tau_var_39_1 + r[39,1] * exp_tau_var_39_2
   + r[39,1] * exp_tau_var_39_3 + r[39,1] * exp_tau_var_39_4
   + r[39,1] * exp_tau_var_39_5 + r[39,1] * exp_tau_var_39_6
   + r[39,1] * exp_tau_var_39_7 ] = 0
 softmax_39_2: - exp_tau_var_39_2 + [ r[39,2] * exp_tau_var_39_0
   + r[39,2] * exp_tau_var_39_1 + r[39,2] * exp_tau_var_39_2
   + r[39,2] * exp_tau_var_39_3 + r[39,2] * exp_tau_var_39_4
   + r[39,2] * exp_tau_var_39_5 + r[39,2] * exp_tau_var_39_6
   + r[39,2] * exp_tau_var_39_7 ] = 0
 softmax_39_3: - exp_tau_var_39_3 + [ r[39,3] * exp_tau_var_39_0
   + r[39,3] * exp_tau_var_39_1 + r[39,3] * exp_tau_var_39_2
   + r[39,3] * exp_tau_var_39_3 + r[39,3] * exp_tau_var_39_4
   + r[39,3] * exp_tau_var_39_5 + r[39,3] * exp_tau_var_39_6
   + r[39,3] * exp_tau_var_39_7 ] = 0
 softmax_39_4: - exp_tau_var_39_4 + [ r[39,4] * exp_tau_var_39_0
   + r[39,4] * exp_tau_var_39_1 + r[39,4] * exp_tau_var_39_2
   + r[39,4] * exp_tau_var_39_3 + r[39,4] * exp_tau_var_39_4
   + r[39,4] * exp_tau_var_39_5 + r[39,4] * exp_tau_var_39_6
   + r[39,4] * exp_tau_var_39_7 ] = 0
 softmax_39_5: - exp_tau_var_39_5 + [ r[39,5] * exp_tau_var_39_0
   + r[39,5] * exp_tau_var_39_1 + r[39,5] * exp_tau_var_39_2
   + r[39,5] * exp_tau_var_39_3 + r[39,5] * exp_tau_var_39_4
   + r[39,5] * exp_tau_var_39_5 + r[39,5] * exp_tau_var_39_6
   + r[39,5] * exp_tau_var_39_7 ] = 0
 softmax_39_6: - exp_tau_var_39_6 + [ r[39,6] * exp_tau_var_39_0
   + r[39,6] * exp_tau_var_39_1 + r[39,6] * exp_tau_var_39_2
   + r[39,6] * exp_tau_var_39_3 + r[39,6] * exp_tau_var_39_4
   + r[39,6] * exp_tau_var_39_5 + r[39,6] * exp_tau_var_39_6
   + r[39,6] * exp_tau_var_39_7 ] = 0
 softmax_39_7: - exp_tau_var_39_7 + [ r[39,7] * exp_tau_var_39_0
   + r[39,7] * exp_tau_var_39_1 + r[39,7] * exp_tau_var_39_2
   + r[39,7] * exp_tau_var_39_3 + r[39,7] * exp_tau_var_39_4
   + r[39,7] * exp_tau_var_39_5 + r[39,7] * exp_tau_var_39_6
   + r[39,7] * exp_tau_var_39_7 ] = 0
 softmax_40_0: - exp_tau_var_40_0 + [ r[40,0] * exp_tau_var_40_0
   + r[40,0] * exp_tau_var_40_1 + r[40,0] * exp_tau_var_40_2
   + r[40,0] * exp_tau_var_40_3 + r[40,0] * exp_tau_var_40_4
   + r[40,0] * exp_tau_var_40_5 + r[40,0] * exp_tau_var_40_6
   + r[40,0] * exp_tau_var_40_7 ] = 0
 softmax_40_1: - exp_tau_var_40_1 + [ r[40,1] * exp_tau_var_40_0
   + r[40,1] * exp_tau_var_40_1 + r[40,1] * exp_tau_var_40_2
   + r[40,1] * exp_tau_var_40_3 + r[40,1] * exp_tau_var_40_4
   + r[40,1] * exp_tau_var_40_5 + r[40,1] * exp_tau_var_40_6
   + r[40,1] * exp_tau_var_40_7 ] = 0
 softmax_40_2: - exp_tau_var_40_2 + [ r[40,2] * exp_tau_var_40_0
   + r[40,2] * exp_tau_var_40_1 + r[40,2] * exp_tau_var_40_2
   + r[40,2] * exp_tau_var_40_3 + r[40,2] * exp_tau_var_40_4
   + r[40,2] * exp_tau_var_40_5 + r[40,2] * exp_tau_var_40_6
   + r[40,2] * exp_tau_var_40_7 ] = 0
 softmax_40_3: - exp_tau_var_40_3 + [ r[40,3] * exp_tau_var_40_0
   + r[40,3] * exp_tau_var_40_1 + r[40,3] * exp_tau_var_40_2
   + r[40,3] * exp_tau_var_40_3 + r[40,3] * exp_tau_var_40_4
   + r[40,3] * exp_tau_var_40_5 + r[40,3] * exp_tau_var_40_6
   + r[40,3] * exp_tau_var_40_7 ] = 0
 softmax_40_4: - exp_tau_var_40_4 + [ r[40,4] * exp_tau_var_40_0
   + r[40,4] * exp_tau_var_40_1 + r[40,4] * exp_tau_var_40_2
   + r[40,4] * exp_tau_var_40_3 + r[40,4] * exp_tau_var_40_4
   + r[40,4] * exp_tau_var_40_5 + r[40,4] * exp_tau_var_40_6
   + r[40,4] * exp_tau_var_40_7 ] = 0
 softmax_40_5: - exp_tau_var_40_5 + [ r[40,5] * exp_tau_var_40_0
   + r[40,5] * exp_tau_var_40_1 + r[40,5] * exp_tau_var_40_2
   + r[40,5] * exp_tau_var_40_3 + r[40,5] * exp_tau_var_40_4
   + r[40,5] * exp_tau_var_40_5 + r[40,5] * exp_tau_var_40_6
   + r[40,5] * exp_tau_var_40_7 ] = 0
 softmax_40_6: - exp_tau_var_40_6 + [ r[40,6] * exp_tau_var_40_0
   + r[40,6] * exp_tau_var_40_1 + r[40,6] * exp_tau_var_40_2
   + r[40,6] * exp_tau_var_40_3 + r[40,6] * exp_tau_var_40_4
   + r[40,6] * exp_tau_var_40_5 + r[40,6] * exp_tau_var_40_6
   + r[40,6] * exp_tau_var_40_7 ] = 0
 softmax_40_7: - exp_tau_var_40_7 + [ r[40,7] * exp_tau_var_40_0
   + r[40,7] * exp_tau_var_40_1 + r[40,7] * exp_tau_var_40_2
   + r[40,7] * exp_tau_var_40_3 + r[40,7] * exp_tau_var_40_4
   + r[40,7] * exp_tau_var_40_5 + r[40,7] * exp_tau_var_40_6
   + r[40,7] * exp_tau_var_40_7 ] = 0
 softmax_41_0: - exp_tau_var_41_0 + [ r[41,0] * exp_tau_var_41_0
   + r[41,0] * exp_tau_var_41_1 + r[41,0] * exp_tau_var_41_2
   + r[41,0] * exp_tau_var_41_3 + r[41,0] * exp_tau_var_41_4
   + r[41,0] * exp_tau_var_41_5 + r[41,0] * exp_tau_var_41_6
   + r[41,0] * exp_tau_var_41_7 ] = 0
 softmax_41_1: - exp_tau_var_41_1 + [ r[41,1] * exp_tau_var_41_0
   + r[41,1] * exp_tau_var_41_1 + r[41,1] * exp_tau_var_41_2
   + r[41,1] * exp_tau_var_41_3 + r[41,1] * exp_tau_var_41_4
   + r[41,1] * exp_tau_var_41_5 + r[41,1] * exp_tau_var_41_6
   + r[41,1] * exp_tau_var_41_7 ] = 0
 softmax_41_2: - exp_tau_var_41_2 + [ r[41,2] * exp_tau_var_41_0
   + r[41,2] * exp_tau_var_41_1 + r[41,2] * exp_tau_var_41_2
   + r[41,2] * exp_tau_var_41_3 + r[41,2] * exp_tau_var_41_4
   + r[41,2] * exp_tau_var_41_5 + r[41,2] * exp_tau_var_41_6
   + r[41,2] * exp_tau_var_41_7 ] = 0
 softmax_41_3: - exp_tau_var_41_3 + [ r[41,3] * exp_tau_var_41_0
   + r[41,3] * exp_tau_var_41_1 + r[41,3] * exp_tau_var_41_2
   + r[41,3] * exp_tau_var_41_3 + r[41,3] * exp_tau_var_41_4
   + r[41,3] * exp_tau_var_41_5 + r[41,3] * exp_tau_var_41_6
   + r[41,3] * exp_tau_var_41_7 ] = 0
 softmax_41_4: - exp_tau_var_41_4 + [ r[41,4] * exp_tau_var_41_0
   + r[41,4] * exp_tau_var_41_1 + r[41,4] * exp_tau_var_41_2
   + r[41,4] * exp_tau_var_41_3 + r[41,4] * exp_tau_var_41_4
   + r[41,4] * exp_tau_var_41_5 + r[41,4] * exp_tau_var_41_6
   + r[41,4] * exp_tau_var_41_7 ] = 0
 softmax_41_5: - exp_tau_var_41_5 + [ r[41,5] * exp_tau_var_41_0
   + r[41,5] * exp_tau_var_41_1 + r[41,5] * exp_tau_var_41_2
   + r[41,5] * exp_tau_var_41_3 + r[41,5] * exp_tau_var_41_4
   + r[41,5] * exp_tau_var_41_5 + r[41,5] * exp_tau_var_41_6
   + r[41,5] * exp_tau_var_41_7 ] = 0
 softmax_41_6: - exp_tau_var_41_6 + [ r[41,6] * exp_tau_var_41_0
   + r[41,6] * exp_tau_var_41_1 + r[41,6] * exp_tau_var_41_2
   + r[41,6] * exp_tau_var_41_3 + r[41,6] * exp_tau_var_41_4
   + r[41,6] * exp_tau_var_41_5 + r[41,6] * exp_tau_var_41_6
   + r[41,6] * exp_tau_var_41_7 ] = 0
 softmax_41_7: - exp_tau_var_41_7 + [ r[41,7] * exp_tau_var_41_0
   + r[41,7] * exp_tau_var_41_1 + r[41,7] * exp_tau_var_41_2
   + r[41,7] * exp_tau_var_41_3 + r[41,7] * exp_tau_var_41_4
   + r[41,7] * exp_tau_var_41_5 + r[41,7] * exp_tau_var_41_6
   + r[41,7] * exp_tau_var_41_7 ] = 0
 softmax_42_0: - exp_tau_var_42_0 + [ r[42,0] * exp_tau_var_42_0
   + r[42,0] * exp_tau_var_42_1 + r[42,0] * exp_tau_var_42_2
   + r[42,0] * exp_tau_var_42_3 + r[42,0] * exp_tau_var_42_4
   + r[42,0] * exp_tau_var_42_5 + r[42,0] * exp_tau_var_42_6
   + r[42,0] * exp_tau_var_42_7 ] = 0
 softmax_42_1: - exp_tau_var_42_1 + [ r[42,1] * exp_tau_var_42_0
   + r[42,1] * exp_tau_var_42_1 + r[42,1] * exp_tau_var_42_2
   + r[42,1] * exp_tau_var_42_3 + r[42,1] * exp_tau_var_42_4
   + r[42,1] * exp_tau_var_42_5 + r[42,1] * exp_tau_var_42_6
   + r[42,1] * exp_tau_var_42_7 ] = 0
 softmax_42_2: - exp_tau_var_42_2 + [ r[42,2] * exp_tau_var_42_0
   + r[42,2] * exp_tau_var_42_1 + r[42,2] * exp_tau_var_42_2
   + r[42,2] * exp_tau_var_42_3 + r[42,2] * exp_tau_var_42_4
   + r[42,2] * exp_tau_var_42_5 + r[42,2] * exp_tau_var_42_6
   + r[42,2] * exp_tau_var_42_7 ] = 0
 softmax_42_3: - exp_tau_var_42_3 + [ r[42,3] * exp_tau_var_42_0
   + r[42,3] * exp_tau_var_42_1 + r[42,3] * exp_tau_var_42_2
   + r[42,3] * exp_tau_var_42_3 + r[42,3] * exp_tau_var_42_4
   + r[42,3] * exp_tau_var_42_5 + r[42,3] * exp_tau_var_42_6
   + r[42,3] * exp_tau_var_42_7 ] = 0
 softmax_42_4: - exp_tau_var_42_4 + [ r[42,4] * exp_tau_var_42_0
   + r[42,4] * exp_tau_var_42_1 + r[42,4] * exp_tau_var_42_2
   + r[42,4] * exp_tau_var_42_3 + r[42,4] * exp_tau_var_42_4
   + r[42,4] * exp_tau_var_42_5 + r[42,4] * exp_tau_var_42_6
   + r[42,4] * exp_tau_var_42_7 ] = 0
 softmax_42_5: - exp_tau_var_42_5 + [ r[42,5] * exp_tau_var_42_0
   + r[42,5] * exp_tau_var_42_1 + r[42,5] * exp_tau_var_42_2
   + r[42,5] * exp_tau_var_42_3 + r[42,5] * exp_tau_var_42_4
   + r[42,5] * exp_tau_var_42_5 + r[42,5] * exp_tau_var_42_6
   + r[42,5] * exp_tau_var_42_7 ] = 0
 softmax_42_6: - exp_tau_var_42_6 + [ r[42,6] * exp_tau_var_42_0
   + r[42,6] * exp_tau_var_42_1 + r[42,6] * exp_tau_var_42_2
   + r[42,6] * exp_tau_var_42_3 + r[42,6] * exp_tau_var_42_4
   + r[42,6] * exp_tau_var_42_5 + r[42,6] * exp_tau_var_42_6
   + r[42,6] * exp_tau_var_42_7 ] = 0
 softmax_42_7: - exp_tau_var_42_7 + [ r[42,7] * exp_tau_var_42_0
   + r[42,7] * exp_tau_var_42_1 + r[42,7] * exp_tau_var_42_2
   + r[42,7] * exp_tau_var_42_3 + r[42,7] * exp_tau_var_42_4
   + r[42,7] * exp_tau_var_42_5 + r[42,7] * exp_tau_var_42_6
   + r[42,7] * exp_tau_var_42_7 ] = 0
 softmax_43_0: - exp_tau_var_43_0 + [ r[43,0] * exp_tau_var_43_0
   + r[43,0] * exp_tau_var_43_1 + r[43,0] * exp_tau_var_43_2
   + r[43,0] * exp_tau_var_43_3 + r[43,0] * exp_tau_var_43_4
   + r[43,0] * exp_tau_var_43_5 + r[43,0] * exp_tau_var_43_6
   + r[43,0] * exp_tau_var_43_7 ] = 0
 softmax_43_1: - exp_tau_var_43_1 + [ r[43,1] * exp_tau_var_43_0
   + r[43,1] * exp_tau_var_43_1 + r[43,1] * exp_tau_var_43_2
   + r[43,1] * exp_tau_var_43_3 + r[43,1] * exp_tau_var_43_4
   + r[43,1] * exp_tau_var_43_5 + r[43,1] * exp_tau_var_43_6
   + r[43,1] * exp_tau_var_43_7 ] = 0
 softmax_43_2: - exp_tau_var_43_2 + [ r[43,2] * exp_tau_var_43_0
   + r[43,2] * exp_tau_var_43_1 + r[43,2] * exp_tau_var_43_2
   + r[43,2] * exp_tau_var_43_3 + r[43,2] * exp_tau_var_43_4
   + r[43,2] * exp_tau_var_43_5 + r[43,2] * exp_tau_var_43_6
   + r[43,2] * exp_tau_var_43_7 ] = 0
 softmax_43_3: - exp_tau_var_43_3 + [ r[43,3] * exp_tau_var_43_0
   + r[43,3] * exp_tau_var_43_1 + r[43,3] * exp_tau_var_43_2
   + r[43,3] * exp_tau_var_43_3 + r[43,3] * exp_tau_var_43_4
   + r[43,3] * exp_tau_var_43_5 + r[43,3] * exp_tau_var_43_6
   + r[43,3] * exp_tau_var_43_7 ] = 0
 softmax_43_4: - exp_tau_var_43_4 + [ r[43,4] * exp_tau_var_43_0
   + r[43,4] * exp_tau_var_43_1 + r[43,4] * exp_tau_var_43_2
   + r[43,4] * exp_tau_var_43_3 + r[43,4] * exp_tau_var_43_4
   + r[43,4] * exp_tau_var_43_5 + r[43,4] * exp_tau_var_43_6
   + r[43,4] * exp_tau_var_43_7 ] = 0
 softmax_43_5: - exp_tau_var_43_5 + [ r[43,5] * exp_tau_var_43_0
   + r[43,5] * exp_tau_var_43_1 + r[43,5] * exp_tau_var_43_2
   + r[43,5] * exp_tau_var_43_3 + r[43,5] * exp_tau_var_43_4
   + r[43,5] * exp_tau_var_43_5 + r[43,5] * exp_tau_var_43_6
   + r[43,5] * exp_tau_var_43_7 ] = 0
 softmax_43_6: - exp_tau_var_43_6 + [ r[43,6] * exp_tau_var_43_0
   + r[43,6] * exp_tau_var_43_1 + r[43,6] * exp_tau_var_43_2
   + r[43,6] * exp_tau_var_43_3 + r[43,6] * exp_tau_var_43_4
   + r[43,6] * exp_tau_var_43_5 + r[43,6] * exp_tau_var_43_6
   + r[43,6] * exp_tau_var_43_7 ] = 0
 softmax_43_7: - exp_tau_var_43_7 + [ r[43,7] * exp_tau_var_43_0
   + r[43,7] * exp_tau_var_43_1 + r[43,7] * exp_tau_var_43_2
   + r[43,7] * exp_tau_var_43_3 + r[43,7] * exp_tau_var_43_4
   + r[43,7] * exp_tau_var_43_5 + r[43,7] * exp_tau_var_43_6
   + r[43,7] * exp_tau_var_43_7 ] = 0
 softmax_44_0: - exp_tau_var_44_0 + [ r[44,0] * exp_tau_var_44_0
   + r[44,0] * exp_tau_var_44_1 + r[44,0] * exp_tau_var_44_2
   + r[44,0] * exp_tau_var_44_3 + r[44,0] * exp_tau_var_44_4
   + r[44,0] * exp_tau_var_44_5 + r[44,0] * exp_tau_var_44_6
   + r[44,0] * exp_tau_var_44_7 ] = 0
 softmax_44_1: - exp_tau_var_44_1 + [ r[44,1] * exp_tau_var_44_0
   + r[44,1] * exp_tau_var_44_1 + r[44,1] * exp_tau_var_44_2
   + r[44,1] * exp_tau_var_44_3 + r[44,1] * exp_tau_var_44_4
   + r[44,1] * exp_tau_var_44_5 + r[44,1] * exp_tau_var_44_6
   + r[44,1] * exp_tau_var_44_7 ] = 0
 softmax_44_2: - exp_tau_var_44_2 + [ r[44,2] * exp_tau_var_44_0
   + r[44,2] * exp_tau_var_44_1 + r[44,2] * exp_tau_var_44_2
   + r[44,2] * exp_tau_var_44_3 + r[44,2] * exp_tau_var_44_4
   + r[44,2] * exp_tau_var_44_5 + r[44,2] * exp_tau_var_44_6
   + r[44,2] * exp_tau_var_44_7 ] = 0
 softmax_44_3: - exp_tau_var_44_3 + [ r[44,3] * exp_tau_var_44_0
   + r[44,3] * exp_tau_var_44_1 + r[44,3] * exp_tau_var_44_2
   + r[44,3] * exp_tau_var_44_3 + r[44,3] * exp_tau_var_44_4
   + r[44,3] * exp_tau_var_44_5 + r[44,3] * exp_tau_var_44_6
   + r[44,3] * exp_tau_var_44_7 ] = 0
 softmax_44_4: - exp_tau_var_44_4 + [ r[44,4] * exp_tau_var_44_0
   + r[44,4] * exp_tau_var_44_1 + r[44,4] * exp_tau_var_44_2
   + r[44,4] * exp_tau_var_44_3 + r[44,4] * exp_tau_var_44_4
   + r[44,4] * exp_tau_var_44_5 + r[44,4] * exp_tau_var_44_6
   + r[44,4] * exp_tau_var_44_7 ] = 0
 softmax_44_5: - exp_tau_var_44_5 + [ r[44,5] * exp_tau_var_44_0
   + r[44,5] * exp_tau_var_44_1 + r[44,5] * exp_tau_var_44_2
   + r[44,5] * exp_tau_var_44_3 + r[44,5] * exp_tau_var_44_4
   + r[44,5] * exp_tau_var_44_5 + r[44,5] * exp_tau_var_44_6
   + r[44,5] * exp_tau_var_44_7 ] = 0
 softmax_44_6: - exp_tau_var_44_6 + [ r[44,6] * exp_tau_var_44_0
   + r[44,6] * exp_tau_var_44_1 + r[44,6] * exp_tau_var_44_2
   + r[44,6] * exp_tau_var_44_3 + r[44,6] * exp_tau_var_44_4
   + r[44,6] * exp_tau_var_44_5 + r[44,6] * exp_tau_var_44_6
   + r[44,6] * exp_tau_var_44_7 ] = 0
 softmax_44_7: - exp_tau_var_44_7 + [ r[44,7] * exp_tau_var_44_0
   + r[44,7] * exp_tau_var_44_1 + r[44,7] * exp_tau_var_44_2
   + r[44,7] * exp_tau_var_44_3 + r[44,7] * exp_tau_var_44_4
   + r[44,7] * exp_tau_var_44_5 + r[44,7] * exp_tau_var_44_6
   + r[44,7] * exp_tau_var_44_7 ] = 0
 softmax_45_0: - exp_tau_var_45_0 + [ r[45,0] * exp_tau_var_45_0
   + r[45,0] * exp_tau_var_45_1 + r[45,0] * exp_tau_var_45_2
   + r[45,0] * exp_tau_var_45_3 + r[45,0] * exp_tau_var_45_4
   + r[45,0] * exp_tau_var_45_5 + r[45,0] * exp_tau_var_45_6
   + r[45,0] * exp_tau_var_45_7 ] = 0
 softmax_45_1: - exp_tau_var_45_1 + [ r[45,1] * exp_tau_var_45_0
   + r[45,1] * exp_tau_var_45_1 + r[45,1] * exp_tau_var_45_2
   + r[45,1] * exp_tau_var_45_3 + r[45,1] * exp_tau_var_45_4
   + r[45,1] * exp_tau_var_45_5 + r[45,1] * exp_tau_var_45_6
   + r[45,1] * exp_tau_var_45_7 ] = 0
 softmax_45_2: - exp_tau_var_45_2 + [ r[45,2] * exp_tau_var_45_0
   + r[45,2] * exp_tau_var_45_1 + r[45,2] * exp_tau_var_45_2
   + r[45,2] * exp_tau_var_45_3 + r[45,2] * exp_tau_var_45_4
   + r[45,2] * exp_tau_var_45_5 + r[45,2] * exp_tau_var_45_6
   + r[45,2] * exp_tau_var_45_7 ] = 0
 softmax_45_3: - exp_tau_var_45_3 + [ r[45,3] * exp_tau_var_45_0
   + r[45,3] * exp_tau_var_45_1 + r[45,3] * exp_tau_var_45_2
   + r[45,3] * exp_tau_var_45_3 + r[45,3] * exp_tau_var_45_4
   + r[45,3] * exp_tau_var_45_5 + r[45,3] * exp_tau_var_45_6
   + r[45,3] * exp_tau_var_45_7 ] = 0
 softmax_45_4: - exp_tau_var_45_4 + [ r[45,4] * exp_tau_var_45_0
   + r[45,4] * exp_tau_var_45_1 + r[45,4] * exp_tau_var_45_2
   + r[45,4] * exp_tau_var_45_3 + r[45,4] * exp_tau_var_45_4
   + r[45,4] * exp_tau_var_45_5 + r[45,4] * exp_tau_var_45_6
   + r[45,4] * exp_tau_var_45_7 ] = 0
 softmax_45_5: - exp_tau_var_45_5 + [ r[45,5] * exp_tau_var_45_0
   + r[45,5] * exp_tau_var_45_1 + r[45,5] * exp_tau_var_45_2
   + r[45,5] * exp_tau_var_45_3 + r[45,5] * exp_tau_var_45_4
   + r[45,5] * exp_tau_var_45_5 + r[45,5] * exp_tau_var_45_6
   + r[45,5] * exp_tau_var_45_7 ] = 0
 softmax_45_6: - exp_tau_var_45_6 + [ r[45,6] * exp_tau_var_45_0
   + r[45,6] * exp_tau_var_45_1 + r[45,6] * exp_tau_var_45_2
   + r[45,6] * exp_tau_var_45_3 + r[45,6] * exp_tau_var_45_4
   + r[45,6] * exp_tau_var_45_5 + r[45,6] * exp_tau_var_45_6
   + r[45,6] * exp_tau_var_45_7 ] = 0
 softmax_45_7: - exp_tau_var_45_7 + [ r[45,7] * exp_tau_var_45_0
   + r[45,7] * exp_tau_var_45_1 + r[45,7] * exp_tau_var_45_2
   + r[45,7] * exp_tau_var_45_3 + r[45,7] * exp_tau_var_45_4
   + r[45,7] * exp_tau_var_45_5 + r[45,7] * exp_tau_var_45_6
   + r[45,7] * exp_tau_var_45_7 ] = 0
 softmax_46_0: - exp_tau_var_46_0 + [ r[46,0] * exp_tau_var_46_0
   + r[46,0] * exp_tau_var_46_1 + r[46,0] * exp_tau_var_46_2
   + r[46,0] * exp_tau_var_46_3 + r[46,0] * exp_tau_var_46_4
   + r[46,0] * exp_tau_var_46_5 + r[46,0] * exp_tau_var_46_6
   + r[46,0] * exp_tau_var_46_7 ] = 0
 softmax_46_1: - exp_tau_var_46_1 + [ r[46,1] * exp_tau_var_46_0
   + r[46,1] * exp_tau_var_46_1 + r[46,1] * exp_tau_var_46_2
   + r[46,1] * exp_tau_var_46_3 + r[46,1] * exp_tau_var_46_4
   + r[46,1] * exp_tau_var_46_5 + r[46,1] * exp_tau_var_46_6
   + r[46,1] * exp_tau_var_46_7 ] = 0
 softmax_46_2: - exp_tau_var_46_2 + [ r[46,2] * exp_tau_var_46_0
   + r[46,2] * exp_tau_var_46_1 + r[46,2] * exp_tau_var_46_2
   + r[46,2] * exp_tau_var_46_3 + r[46,2] * exp_tau_var_46_4
   + r[46,2] * exp_tau_var_46_5 + r[46,2] * exp_tau_var_46_6
   + r[46,2] * exp_tau_var_46_7 ] = 0
 softmax_46_3: - exp_tau_var_46_3 + [ r[46,3] * exp_tau_var_46_0
   + r[46,3] * exp_tau_var_46_1 + r[46,3] * exp_tau_var_46_2
   + r[46,3] * exp_tau_var_46_3 + r[46,3] * exp_tau_var_46_4
   + r[46,3] * exp_tau_var_46_5 + r[46,3] * exp_tau_var_46_6
   + r[46,3] * exp_tau_var_46_7 ] = 0
 softmax_46_4: - exp_tau_var_46_4 + [ r[46,4] * exp_tau_var_46_0
   + r[46,4] * exp_tau_var_46_1 + r[46,4] * exp_tau_var_46_2
   + r[46,4] * exp_tau_var_46_3 + r[46,4] * exp_tau_var_46_4
   + r[46,4] * exp_tau_var_46_5 + r[46,4] * exp_tau_var_46_6
   + r[46,4] * exp_tau_var_46_7 ] = 0
 softmax_46_5: - exp_tau_var_46_5 + [ r[46,5] * exp_tau_var_46_0
   + r[46,5] * exp_tau_var_46_1 + r[46,5] * exp_tau_var_46_2
   + r[46,5] * exp_tau_var_46_3 + r[46,5] * exp_tau_var_46_4
   + r[46,5] * exp_tau_var_46_5 + r[46,5] * exp_tau_var_46_6
   + r[46,5] * exp_tau_var_46_7 ] = 0
 softmax_46_6: - exp_tau_var_46_6 + [ r[46,6] * exp_tau_var_46_0
   + r[46,6] * exp_tau_var_46_1 + r[46,6] * exp_tau_var_46_2
   + r[46,6] * exp_tau_var_46_3 + r[46,6] * exp_tau_var_46_4
   + r[46,6] * exp_tau_var_46_5 + r[46,6] * exp_tau_var_46_6
   + r[46,6] * exp_tau_var_46_7 ] = 0
 softmax_46_7: - exp_tau_var_46_7 + [ r[46,7] * exp_tau_var_46_0
   + r[46,7] * exp_tau_var_46_1 + r[46,7] * exp_tau_var_46_2
   + r[46,7] * exp_tau_var_46_3 + r[46,7] * exp_tau_var_46_4
   + r[46,7] * exp_tau_var_46_5 + r[46,7] * exp_tau_var_46_6
   + r[46,7] * exp_tau_var_46_7 ] = 0
 softmax_47_0: - exp_tau_var_47_0 + [ r[47,0] * exp_tau_var_47_0
   + r[47,0] * exp_tau_var_47_1 + r[47,0] * exp_tau_var_47_2
   + r[47,0] * exp_tau_var_47_3 + r[47,0] * exp_tau_var_47_4
   + r[47,0] * exp_tau_var_47_5 + r[47,0] * exp_tau_var_47_6
   + r[47,0] * exp_tau_var_47_7 ] = 0
 softmax_47_1: - exp_tau_var_47_1 + [ r[47,1] * exp_tau_var_47_0
   + r[47,1] * exp_tau_var_47_1 + r[47,1] * exp_tau_var_47_2
   + r[47,1] * exp_tau_var_47_3 + r[47,1] * exp_tau_var_47_4
   + r[47,1] * exp_tau_var_47_5 + r[47,1] * exp_tau_var_47_6
   + r[47,1] * exp_tau_var_47_7 ] = 0
 softmax_47_2: - exp_tau_var_47_2 + [ r[47,2] * exp_tau_var_47_0
   + r[47,2] * exp_tau_var_47_1 + r[47,2] * exp_tau_var_47_2
   + r[47,2] * exp_tau_var_47_3 + r[47,2] * exp_tau_var_47_4
   + r[47,2] * exp_tau_var_47_5 + r[47,2] * exp_tau_var_47_6
   + r[47,2] * exp_tau_var_47_7 ] = 0
 softmax_47_3: - exp_tau_var_47_3 + [ r[47,3] * exp_tau_var_47_0
   + r[47,3] * exp_tau_var_47_1 + r[47,3] * exp_tau_var_47_2
   + r[47,3] * exp_tau_var_47_3 + r[47,3] * exp_tau_var_47_4
   + r[47,3] * exp_tau_var_47_5 + r[47,3] * exp_tau_var_47_6
   + r[47,3] * exp_tau_var_47_7 ] = 0
 softmax_47_4: - exp_tau_var_47_4 + [ r[47,4] * exp_tau_var_47_0
   + r[47,4] * exp_tau_var_47_1 + r[47,4] * exp_tau_var_47_2
   + r[47,4] * exp_tau_var_47_3 + r[47,4] * exp_tau_var_47_4
   + r[47,4] * exp_tau_var_47_5 + r[47,4] * exp_tau_var_47_6
   + r[47,4] * exp_tau_var_47_7 ] = 0
 softmax_47_5: - exp_tau_var_47_5 + [ r[47,5] * exp_tau_var_47_0
   + r[47,5] * exp_tau_var_47_1 + r[47,5] * exp_tau_var_47_2
   + r[47,5] * exp_tau_var_47_3 + r[47,5] * exp_tau_var_47_4
   + r[47,5] * exp_tau_var_47_5 + r[47,5] * exp_tau_var_47_6
   + r[47,5] * exp_tau_var_47_7 ] = 0
 softmax_47_6: - exp_tau_var_47_6 + [ r[47,6] * exp_tau_var_47_0
   + r[47,6] * exp_tau_var_47_1 + r[47,6] * exp_tau_var_47_2
   + r[47,6] * exp_tau_var_47_3 + r[47,6] * exp_tau_var_47_4
   + r[47,6] * exp_tau_var_47_5 + r[47,6] * exp_tau_var_47_6
   + r[47,6] * exp_tau_var_47_7 ] = 0
 softmax_47_7: - exp_tau_var_47_7 + [ r[47,7] * exp_tau_var_47_0
   + r[47,7] * exp_tau_var_47_1 + r[47,7] * exp_tau_var_47_2
   + r[47,7] * exp_tau_var_47_3 + r[47,7] * exp_tau_var_47_4
   + r[47,7] * exp_tau_var_47_5 + r[47,7] * exp_tau_var_47_6
   + r[47,7] * exp_tau_var_47_7 ] = 0
 softmax_48_0: - exp_tau_var_48_0 + [ r[48,0] * exp_tau_var_48_0
   + r[48,0] * exp_tau_var_48_1 + r[48,0] * exp_tau_var_48_2
   + r[48,0] * exp_tau_var_48_3 + r[48,0] * exp_tau_var_48_4
   + r[48,0] * exp_tau_var_48_5 + r[48,0] * exp_tau_var_48_6
   + r[48,0] * exp_tau_var_48_7 ] = 0
 softmax_48_1: - exp_tau_var_48_1 + [ r[48,1] * exp_tau_var_48_0
   + r[48,1] * exp_tau_var_48_1 + r[48,1] * exp_tau_var_48_2
   + r[48,1] * exp_tau_var_48_3 + r[48,1] * exp_tau_var_48_4
   + r[48,1] * exp_tau_var_48_5 + r[48,1] * exp_tau_var_48_6
   + r[48,1] * exp_tau_var_48_7 ] = 0
 softmax_48_2: - exp_tau_var_48_2 + [ r[48,2] * exp_tau_var_48_0
   + r[48,2] * exp_tau_var_48_1 + r[48,2] * exp_tau_var_48_2
   + r[48,2] * exp_tau_var_48_3 + r[48,2] * exp_tau_var_48_4
   + r[48,2] * exp_tau_var_48_5 + r[48,2] * exp_tau_var_48_6
   + r[48,2] * exp_tau_var_48_7 ] = 0
 softmax_48_3: - exp_tau_var_48_3 + [ r[48,3] * exp_tau_var_48_0
   + r[48,3] * exp_tau_var_48_1 + r[48,3] * exp_tau_var_48_2
   + r[48,3] * exp_tau_var_48_3 + r[48,3] * exp_tau_var_48_4
   + r[48,3] * exp_tau_var_48_5 + r[48,3] * exp_tau_var_48_6
   + r[48,3] * exp_tau_var_48_7 ] = 0
 softmax_48_4: - exp_tau_var_48_4 + [ r[48,4] * exp_tau_var_48_0
   + r[48,4] * exp_tau_var_48_1 + r[48,4] * exp_tau_var_48_2
   + r[48,4] * exp_tau_var_48_3 + r[48,4] * exp_tau_var_48_4
   + r[48,4] * exp_tau_var_48_5 + r[48,4] * exp_tau_var_48_6
   + r[48,4] * exp_tau_var_48_7 ] = 0
 softmax_48_5: - exp_tau_var_48_5 + [ r[48,5] * exp_tau_var_48_0
   + r[48,5] * exp_tau_var_48_1 + r[48,5] * exp_tau_var_48_2
   + r[48,5] * exp_tau_var_48_3 + r[48,5] * exp_tau_var_48_4
   + r[48,5] * exp_tau_var_48_5 + r[48,5] * exp_tau_var_48_6
   + r[48,5] * exp_tau_var_48_7 ] = 0
 softmax_48_6: - exp_tau_var_48_6 + [ r[48,6] * exp_tau_var_48_0
   + r[48,6] * exp_tau_var_48_1 + r[48,6] * exp_tau_var_48_2
   + r[48,6] * exp_tau_var_48_3 + r[48,6] * exp_tau_var_48_4
   + r[48,6] * exp_tau_var_48_5 + r[48,6] * exp_tau_var_48_6
   + r[48,6] * exp_tau_var_48_7 ] = 0
 softmax_48_7: - exp_tau_var_48_7 + [ r[48,7] * exp_tau_var_48_0
   + r[48,7] * exp_tau_var_48_1 + r[48,7] * exp_tau_var_48_2
   + r[48,7] * exp_tau_var_48_3 + r[48,7] * exp_tau_var_48_4
   + r[48,7] * exp_tau_var_48_5 + r[48,7] * exp_tau_var_48_6
   + r[48,7] * exp_tau_var_48_7 ] = 0
 softmax_49_0: - exp_tau_var_49_0 + [ r[49,0] * exp_tau_var_49_0
   + r[49,0] * exp_tau_var_49_1 + r[49,0] * exp_tau_var_49_2
   + r[49,0] * exp_tau_var_49_3 + r[49,0] * exp_tau_var_49_4
   + r[49,0] * exp_tau_var_49_5 + r[49,0] * exp_tau_var_49_6
   + r[49,0] * exp_tau_var_49_7 ] = 0
 softmax_49_1: - exp_tau_var_49_1 + [ r[49,1] * exp_tau_var_49_0
   + r[49,1] * exp_tau_var_49_1 + r[49,1] * exp_tau_var_49_2
   + r[49,1] * exp_tau_var_49_3 + r[49,1] * exp_tau_var_49_4
   + r[49,1] * exp_tau_var_49_5 + r[49,1] * exp_tau_var_49_6
   + r[49,1] * exp_tau_var_49_7 ] = 0
 softmax_49_2: - exp_tau_var_49_2 + [ r[49,2] * exp_tau_var_49_0
   + r[49,2] * exp_tau_var_49_1 + r[49,2] * exp_tau_var_49_2
   + r[49,2] * exp_tau_var_49_3 + r[49,2] * exp_tau_var_49_4
   + r[49,2] * exp_tau_var_49_5 + r[49,2] * exp_tau_var_49_6
   + r[49,2] * exp_tau_var_49_7 ] = 0
 softmax_49_3: - exp_tau_var_49_3 + [ r[49,3] * exp_tau_var_49_0
   + r[49,3] * exp_tau_var_49_1 + r[49,3] * exp_tau_var_49_2
   + r[49,3] * exp_tau_var_49_3 + r[49,3] * exp_tau_var_49_4
   + r[49,3] * exp_tau_var_49_5 + r[49,3] * exp_tau_var_49_6
   + r[49,3] * exp_tau_var_49_7 ] = 0
 softmax_49_4: - exp_tau_var_49_4 + [ r[49,4] * exp_tau_var_49_0
   + r[49,4] * exp_tau_var_49_1 + r[49,4] * exp_tau_var_49_2
   + r[49,4] * exp_tau_var_49_3 + r[49,4] * exp_tau_var_49_4
   + r[49,4] * exp_tau_var_49_5 + r[49,4] * exp_tau_var_49_6
   + r[49,4] * exp_tau_var_49_7 ] = 0
 softmax_49_5: - exp_tau_var_49_5 + [ r[49,5] * exp_tau_var_49_0
   + r[49,5] * exp_tau_var_49_1 + r[49,5] * exp_tau_var_49_2
   + r[49,5] * exp_tau_var_49_3 + r[49,5] * exp_tau_var_49_4
   + r[49,5] * exp_tau_var_49_5 + r[49,5] * exp_tau_var_49_6
   + r[49,5] * exp_tau_var_49_7 ] = 0
 softmax_49_6: - exp_tau_var_49_6 + [ r[49,6] * exp_tau_var_49_0
   + r[49,6] * exp_tau_var_49_1 + r[49,6] * exp_tau_var_49_2
   + r[49,6] * exp_tau_var_49_3 + r[49,6] * exp_tau_var_49_4
   + r[49,6] * exp_tau_var_49_5 + r[49,6] * exp_tau_var_49_6
   + r[49,6] * exp_tau_var_49_7 ] = 0
 softmax_49_7: - exp_tau_var_49_7 + [ r[49,7] * exp_tau_var_49_0
   + r[49,7] * exp_tau_var_49_1 + r[49,7] * exp_tau_var_49_2
   + r[49,7] * exp_tau_var_49_3 + r[49,7] * exp_tau_var_49_4
   + r[49,7] * exp_tau_var_49_5 + r[49,7] * exp_tau_var_49_6
   + r[49,7] * exp_tau_var_49_7 ] = 0
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
 GC635: R[25,0] = 1 -> r[25,0] - max_r_helper = 0
 GC636: R[25,1] = 1 -> r[25,1] - max_r_helper = 0
 GC637: R[25,2] = 1 -> r[25,2] - max_r_helper = 0
 GC638: R[25,3] = 1 -> r[25,3] - max_r_helper = 0
 GC639: R[25,4] = 1 -> r[25,4] - max_r_helper = 0
 GC640: R[25,5] = 1 -> r[25,5] - max_r_helper = 0
 GC641: R[25,6] = 1 -> r[25,6] - max_r_helper = 0
 GC642: R[25,7] = 1 -> r[25,7] - max_r_helper = 0
 GC660: R[26,0] = 1 -> r[26,0] - max_r_helper = 0
 GC661: R[26,1] = 1 -> r[26,1] - max_r_helper = 0
 GC662: R[26,2] = 1 -> r[26,2] - max_r_helper = 0
 GC663: R[26,3] = 1 -> r[26,3] - max_r_helper = 0
 GC664: R[26,4] = 1 -> r[26,4] - max_r_helper = 0
 GC665: R[26,5] = 1 -> r[26,5] - max_r_helper = 0
 GC666: R[26,6] = 1 -> r[26,6] - max_r_helper = 0
 GC667: R[26,7] = 1 -> r[26,7] - max_r_helper = 0
 GC685: R[27,0] = 1 -> r[27,0] - max_r_helper = 0
 GC686: R[27,1] = 1 -> r[27,1] - max_r_helper = 0
 GC687: R[27,2] = 1 -> r[27,2] - max_r_helper = 0
 GC688: R[27,3] = 1 -> r[27,3] - max_r_helper = 0
 GC689: R[27,4] = 1 -> r[27,4] - max_r_helper = 0
 GC690: R[27,5] = 1 -> r[27,5] - max_r_helper = 0
 GC691: R[27,6] = 1 -> r[27,6] - max_r_helper = 0
 GC692: R[27,7] = 1 -> r[27,7] - max_r_helper = 0
 GC710: R[28,0] = 1 -> r[28,0] - max_r_helper = 0
 GC711: R[28,1] = 1 -> r[28,1] - max_r_helper = 0
 GC712: R[28,2] = 1 -> r[28,2] - max_r_helper = 0
 GC713: R[28,3] = 1 -> r[28,3] - max_r_helper = 0
 GC714: R[28,4] = 1 -> r[28,4] - max_r_helper = 0
 GC715: R[28,5] = 1 -> r[28,5] - max_r_helper = 0
 GC716: R[28,6] = 1 -> r[28,6] - max_r_helper = 0
 GC717: R[28,7] = 1 -> r[28,7] - max_r_helper = 0
 GC735: R[29,0] = 1 -> r[29,0] - max_r_helper = 0
 GC736: R[29,1] = 1 -> r[29,1] - max_r_helper = 0
 GC737: R[29,2] = 1 -> r[29,2] - max_r_helper = 0
 GC738: R[29,3] = 1 -> r[29,3] - max_r_helper = 0
 GC739: R[29,4] = 1 -> r[29,4] - max_r_helper = 0
 GC740: R[29,5] = 1 -> r[29,5] - max_r_helper = 0
 GC741: R[29,6] = 1 -> r[29,6] - max_r_helper = 0
 GC742: R[29,7] = 1 -> r[29,7] - max_r_helper = 0
 GC760: R[30,0] = 1 -> r[30,0] - max_r_helper = 0
 GC761: R[30,1] = 1 -> r[30,1] - max_r_helper = 0
 GC762: R[30,2] = 1 -> r[30,2] - max_r_helper = 0
 GC763: R[30,3] = 1 -> r[30,3] - max_r_helper = 0
 GC764: R[30,4] = 1 -> r[30,4] - max_r_helper = 0
 GC765: R[30,5] = 1 -> r[30,5] - max_r_helper = 0
 GC766: R[30,6] = 1 -> r[30,6] - max_r_helper = 0
 GC767: R[30,7] = 1 -> r[30,7] - max_r_helper = 0
 GC785: R[31,0] = 1 -> r[31,0] - max_r_helper = 0
 GC786: R[31,1] = 1 -> r[31,1] - max_r_helper = 0
 GC787: R[31,2] = 1 -> r[31,2] - max_r_helper = 0
 GC788: R[31,3] = 1 -> r[31,3] - max_r_helper = 0
 GC789: R[31,4] = 1 -> r[31,4] - max_r_helper = 0
 GC790: R[31,5] = 1 -> r[31,5] - max_r_helper = 0
 GC791: R[31,6] = 1 -> r[31,6] - max_r_helper = 0
 GC792: R[31,7] = 1 -> r[31,7] - max_r_helper = 0
 GC810: R[32,0] = 1 -> r[32,0] - max_r_helper = 0
 GC811: R[32,1] = 1 -> r[32,1] - max_r_helper = 0
 GC812: R[32,2] = 1 -> r[32,2] - max_r_helper = 0
 GC813: R[32,3] = 1 -> r[32,3] - max_r_helper = 0
 GC814: R[32,4] = 1 -> r[32,4] - max_r_helper = 0
 GC815: R[32,5] = 1 -> r[32,5] - max_r_helper = 0
 GC816: R[32,6] = 1 -> r[32,6] - max_r_helper = 0
 GC817: R[32,7] = 1 -> r[32,7] - max_r_helper = 0
 GC835: R[33,0] = 1 -> r[33,0] - max_r_helper = 0
 GC836: R[33,1] = 1 -> r[33,1] - max_r_helper = 0
 GC837: R[33,2] = 1 -> r[33,2] - max_r_helper = 0
 GC838: R[33,3] = 1 -> r[33,3] - max_r_helper = 0
 GC839: R[33,4] = 1 -> r[33,4] - max_r_helper = 0
 GC840: R[33,5] = 1 -> r[33,5] - max_r_helper = 0
 GC841: R[33,6] = 1 -> r[33,6] - max_r_helper = 0
 GC842: R[33,7] = 1 -> r[33,7] - max_r_helper = 0
 GC860: R[34,0] = 1 -> r[34,0] - max_r_helper = 0
 GC861: R[34,1] = 1 -> r[34,1] - max_r_helper = 0
 GC862: R[34,2] = 1 -> r[34,2] - max_r_helper = 0
 GC863: R[34,3] = 1 -> r[34,3] - max_r_helper = 0
 GC864: R[34,4] = 1 -> r[34,4] - max_r_helper = 0
 GC865: R[34,5] = 1 -> r[34,5] - max_r_helper = 0
 GC866: R[34,6] = 1 -> r[34,6] - max_r_helper = 0
 GC867: R[34,7] = 1 -> r[34,7] - max_r_helper = 0
 GC885: R[35,0] = 1 -> r[35,0] - max_r_helper = 0
 GC886: R[35,1] = 1 -> r[35,1] - max_r_helper = 0
 GC887: R[35,2] = 1 -> r[35,2] - max_r_helper = 0
 GC888: R[35,3] = 1 -> r[35,3] - max_r_helper = 0
 GC889: R[35,4] = 1 -> r[35,4] - max_r_helper = 0
 GC890: R[35,5] = 1 -> r[35,5] - max_r_helper = 0
 GC891: R[35,6] = 1 -> r[35,6] - max_r_helper = 0
 GC892: R[35,7] = 1 -> r[35,7] - max_r_helper = 0
 GC910: R[36,0] = 1 -> r[36,0] - max_r_helper = 0
 GC911: R[36,1] = 1 -> r[36,1] - max_r_helper = 0
 GC912: R[36,2] = 1 -> r[36,2] - max_r_helper = 0
 GC913: R[36,3] = 1 -> r[36,3] - max_r_helper = 0
 GC914: R[36,4] = 1 -> r[36,4] - max_r_helper = 0
 GC915: R[36,5] = 1 -> r[36,5] - max_r_helper = 0
 GC916: R[36,6] = 1 -> r[36,6] - max_r_helper = 0
 GC917: R[36,7] = 1 -> r[36,7] - max_r_helper = 0
 GC935: R[37,0] = 1 -> r[37,0] - max_r_helper = 0
 GC936: R[37,1] = 1 -> r[37,1] - max_r_helper = 0
 GC937: R[37,2] = 1 -> r[37,2] - max_r_helper = 0
 GC938: R[37,3] = 1 -> r[37,3] - max_r_helper = 0
 GC939: R[37,4] = 1 -> r[37,4] - max_r_helper = 0
 GC940: R[37,5] = 1 -> r[37,5] - max_r_helper = 0
 GC941: R[37,6] = 1 -> r[37,6] - max_r_helper = 0
 GC942: R[37,7] = 1 -> r[37,7] - max_r_helper = 0
 GC960: R[38,0] = 1 -> r[38,0] - max_r_helper = 0
 GC961: R[38,1] = 1 -> r[38,1] - max_r_helper = 0
 GC962: R[38,2] = 1 -> r[38,2] - max_r_helper = 0
 GC963: R[38,3] = 1 -> r[38,3] - max_r_helper = 0
 GC964: R[38,4] = 1 -> r[38,4] - max_r_helper = 0
 GC965: R[38,5] = 1 -> r[38,5] - max_r_helper = 0
 GC966: R[38,6] = 1 -> r[38,6] - max_r_helper = 0
 GC967: R[38,7] = 1 -> r[38,7] - max_r_helper = 0
 GC985: R[39,0] = 1 -> r[39,0] - max_r_helper = 0
 GC986: R[39,1] = 1 -> r[39,1] - max_r_helper = 0
 GC987: R[39,2] = 1 -> r[39,2] - max_r_helper = 0
 GC988: R[39,3] = 1 -> r[39,3] - max_r_helper = 0
 GC989: R[39,4] = 1 -> r[39,4] - max_r_helper = 0
 GC990: R[39,5] = 1 -> r[39,5] - max_r_helper = 0
 GC991: R[39,6] = 1 -> r[39,6] - max_r_helper = 0
 GC992: R[39,7] = 1 -> r[39,7] - max_r_helper = 0
 GC1010: R[40,0] = 1 -> r[40,0] - max_r_helper = 0
 GC1011: R[40,1] = 1 -> r[40,1] - max_r_helper = 0
 GC1012: R[40,2] = 1 -> r[40,2] - max_r_helper = 0
 GC1013: R[40,3] = 1 -> r[40,3] - max_r_helper = 0
 GC1014: R[40,4] = 1 -> r[40,4] - max_r_helper = 0
 GC1015: R[40,5] = 1 -> r[40,5] - max_r_helper = 0
 GC1016: R[40,6] = 1 -> r[40,6] - max_r_helper = 0
 GC1017: R[40,7] = 1 -> r[40,7] - max_r_helper = 0
 GC1035: R[41,0] = 1 -> r[41,0] - max_r_helper = 0
 GC1036: R[41,1] = 1 -> r[41,1] - max_r_helper = 0
 GC1037: R[41,2] = 1 -> r[41,2] - max_r_helper = 0
 GC1038: R[41,3] = 1 -> r[41,3] - max_r_helper = 0
 GC1039: R[41,4] = 1 -> r[41,4] - max_r_helper = 0
 GC1040: R[41,5] = 1 -> r[41,5] - max_r_helper = 0
 GC1041: R[41,6] = 1 -> r[41,6] - max_r_helper = 0
 GC1042: R[41,7] = 1 -> r[41,7] - max_r_helper = 0
 GC1060: R[42,0] = 1 -> r[42,0] - max_r_helper = 0
 GC1061: R[42,1] = 1 -> r[42,1] - max_r_helper = 0
 GC1062: R[42,2] = 1 -> r[42,2] - max_r_helper = 0
 GC1063: R[42,3] = 1 -> r[42,3] - max_r_helper = 0
 GC1064: R[42,4] = 1 -> r[42,4] - max_r_helper = 0
 GC1065: R[42,5] = 1 -> r[42,5] - max_r_helper = 0
 GC1066: R[42,6] = 1 -> r[42,6] - max_r_helper = 0
 GC1067: R[42,7] = 1 -> r[42,7] - max_r_helper = 0
 GC1085: R[43,0] = 1 -> r[43,0] - max_r_helper = 0
 GC1086: R[43,1] = 1 -> r[43,1] - max_r_helper = 0
 GC1087: R[43,2] = 1 -> r[43,2] - max_r_helper = 0
 GC1088: R[43,3] = 1 -> r[43,3] - max_r_helper = 0
 GC1089: R[43,4] = 1 -> r[43,4] - max_r_helper = 0
 GC1090: R[43,5] = 1 -> r[43,5] - max_r_helper = 0
 GC1091: R[43,6] = 1 -> r[43,6] - max_r_helper = 0
 GC1092: R[43,7] = 1 -> r[43,7] - max_r_helper = 0
 GC1110: R[44,0] = 1 -> r[44,0] - max_r_helper = 0
 GC1111: R[44,1] = 1 -> r[44,1] - max_r_helper = 0
 GC1112: R[44,2] = 1 -> r[44,2] - max_r_helper = 0
 GC1113: R[44,3] = 1 -> r[44,3] - max_r_helper = 0
 GC1114: R[44,4] = 1 -> r[44,4] - max_r_helper = 0
 GC1115: R[44,5] = 1 -> r[44,5] - max_r_helper = 0
 GC1116: R[44,6] = 1 -> r[44,6] - max_r_helper = 0
 GC1117: R[44,7] = 1 -> r[44,7] - max_r_helper = 0
 GC1135: R[45,0] = 1 -> r[45,0] - max_r_helper = 0
 GC1136: R[45,1] = 1 -> r[45,1] - max_r_helper = 0
 GC1137: R[45,2] = 1 -> r[45,2] - max_r_helper = 0
 GC1138: R[45,3] = 1 -> r[45,3] - max_r_helper = 0
 GC1139: R[45,4] = 1 -> r[45,4] - max_r_helper = 0
 GC1140: R[45,5] = 1 -> r[45,5] - max_r_helper = 0
 GC1141: R[45,6] = 1 -> r[45,6] - max_r_helper = 0
 GC1142: R[45,7] = 1 -> r[45,7] - max_r_helper = 0
 GC1160: R[46,0] = 1 -> r[46,0] - max_r_helper = 0
 GC1161: R[46,1] = 1 -> r[46,1] - max_r_helper = 0
 GC1162: R[46,2] = 1 -> r[46,2] - max_r_helper = 0
 GC1163: R[46,3] = 1 -> r[46,3] - max_r_helper = 0
 GC1164: R[46,4] = 1 -> r[46,4] - max_r_helper = 0
 GC1165: R[46,5] = 1 -> r[46,5] - max_r_helper = 0
 GC1166: R[46,6] = 1 -> r[46,6] - max_r_helper = 0
 GC1167: R[46,7] = 1 -> r[46,7] - max_r_helper = 0
 GC1185: R[47,0] = 1 -> r[47,0] - max_r_helper = 0
 GC1186: R[47,1] = 1 -> r[47,1] - max_r_helper = 0
 GC1187: R[47,2] = 1 -> r[47,2] - max_r_helper = 0
 GC1188: R[47,3] = 1 -> r[47,3] - max_r_helper = 0
 GC1189: R[47,4] = 1 -> r[47,4] - max_r_helper = 0
 GC1190: R[47,5] = 1 -> r[47,5] - max_r_helper = 0
 GC1191: R[47,6] = 1 -> r[47,6] - max_r_helper = 0
 GC1192: R[47,7] = 1 -> r[47,7] - max_r_helper = 0
 GC1210: R[48,0] = 1 -> r[48,0] - max_r_helper = 0
 GC1211: R[48,1] = 1 -> r[48,1] - max_r_helper = 0
 GC1212: R[48,2] = 1 -> r[48,2] - max_r_helper = 0
 GC1213: R[48,3] = 1 -> r[48,3] - max_r_helper = 0
 GC1214: R[48,4] = 1 -> r[48,4] - max_r_helper = 0
 GC1215: R[48,5] = 1 -> r[48,5] - max_r_helper = 0
 GC1216: R[48,6] = 1 -> r[48,6] - max_r_helper = 0
 GC1217: R[48,7] = 1 -> r[48,7] - max_r_helper = 0
 GC1235: R[49,0] = 1 -> r[49,0] - max_r_helper = 0
 GC1236: R[49,1] = 1 -> r[49,1] - max_r_helper = 0
 GC1237: R[49,2] = 1 -> r[49,2] - max_r_helper = 0
 GC1238: R[49,3] = 1 -> r[49,3] - max_r_helper = 0
 GC1239: R[49,4] = 1 -> r[49,4] - max_r_helper = 0
 GC1240: R[49,5] = 1 -> r[49,5] - max_r_helper = 0
 GC1241: R[49,6] = 1 -> r[49,6] - max_r_helper = 0
 GC1242: R[49,7] = 1 -> r[49,7] - max_r_helper = 0
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
 Fraction_for_second_order_amount[25] <= 1
 Fraction_for_second_order_amount[26] <= 1
 Fraction_for_second_order_amount[27] <= 1
 Fraction_for_second_order_amount[28] <= 1
 Fraction_for_second_order_amount[29] <= 1
 Fraction_for_second_order_amount[30] <= 1
 Fraction_for_second_order_amount[31] <= 1
 Fraction_for_second_order_amount[32] <= 1
 Fraction_for_second_order_amount[33] <= 1
 Fraction_for_second_order_amount[34] <= 1
 Fraction_for_second_order_amount[35] <= 1
 Fraction_for_second_order_amount[36] <= 1
 Fraction_for_second_order_amount[37] <= 1
 Fraction_for_second_order_amount[38] <= 1
 Fraction_for_second_order_amount[39] <= 1
 Fraction_for_second_order_amount[40] <= 1
 Fraction_for_second_order_amount[41] <= 1
 Fraction_for_second_order_amount[42] <= 1
 Fraction_for_second_order_amount[43] <= 1
 Fraction_for_second_order_amount[44] <= 1
 Fraction_for_second_order_amount[45] <= 1
 Fraction_for_second_order_amount[46] <= 1
 Fraction_for_second_order_amount[47] <= 1
 Fraction_for_second_order_amount[48] <= 1
 Fraction_for_second_order_amount[49] <= 1
 Q0_var[0] <= 2029.416054007841
 Q0_var[1] <= 2029.416054007841
 Q0_var[2] <= 2029.416054007841
 Q0_var[3] <= 2029.416054007841
 Q0_var[4] <= 2029.416054007841
 Q0_var[5] <= 2029.416054007841
 Q0_var[6] <= 2029.416054007841
 Q0_var[7] <= 2029.416054007841
 Q0_var[8] <= 2029.416054007841
 Q0_var[9] <= 2029.416054007841
 Q0_var[10] <= 2029.416054007841
 Q0_var[11] <= 2029.416054007841
 Q0_var[12] <= 2029.416054007841
 Q0_var[13] <= 2029.416054007841
 Q0_var[14] <= 2029.416054007841
 Q0_var[15] <= 2029.416054007841
 Q0_var[16] <= 2029.416054007841
 Q0_var[17] <= 2029.416054007841
 Q0_var[18] <= 2029.416054007841
 Q0_var[19] <= 2029.416054007841
 Q0_var[20] <= 2029.416054007841
 Q0_var[21] <= 2029.416054007841
 Q0_var[22] <= 2029.416054007841
 Q0_var[23] <= 2029.416054007841
 Q0_var[24] <= 2029.416054007841
 Q0_var[25] <= 2029.416054007841
 Q0_var[26] <= 2029.416054007841
 Q0_var[27] <= 2029.416054007841
 Q0_var[28] <= 2029.416054007841
 Q0_var[29] <= 2029.416054007841
 Q0_var[30] <= 2029.416054007841
 Q0_var[31] <= 2029.416054007841
 Q0_var[32] <= 2029.416054007841
 Q0_var[33] <= 2029.416054007841
 Q0_var[34] <= 2029.416054007841
 Q0_var[35] <= 2029.416054007841
 Q0_var[36] <= 2029.416054007841
 Q0_var[37] <= 2029.416054007841
 Q0_var[38] <= 2029.416054007841
 Q0_var[39] <= 2029.416054007841
 Q0_var[40] <= 2029.416054007841
 Q0_var[41] <= 2029.416054007841
 Q0_var[42] <= 2029.416054007841
 Q0_var[43] <= 2029.416054007841
 Q0_var[44] <= 2029.416054007841
 Q0_var[45] <= 2029.416054007841
 Q0_var[46] <= 2029.416054007841
 Q0_var[47] <= 2029.416054007841
 Q0_var[48] <= 2029.416054007841
 Q0_var[49] <= 2029.416054007841
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
 tau[25,0] free
 tau[25,1] free
 tau[25,2] free
 tau[25,3] free
 tau[25,4] free
 tau[25,5] free
 tau[25,6] free
 tau[25,7] free
 tau[26,0] free
 tau[26,1] free
 tau[26,2] free
 tau[26,3] free
 tau[26,4] free
 tau[26,5] free
 tau[26,6] free
 tau[26,7] free
 tau[27,0] free
 tau[27,1] free
 tau[27,2] free
 tau[27,3] free
 tau[27,4] free
 tau[27,5] free
 tau[27,6] free
 tau[27,7] free
 tau[28,0] free
 tau[28,1] free
 tau[28,2] free
 tau[28,3] free
 tau[28,4] free
 tau[28,5] free
 tau[28,6] free
 tau[28,7] free
 tau[29,0] free
 tau[29,1] free
 tau[29,2] free
 tau[29,3] free
 tau[29,4] free
 tau[29,5] free
 tau[29,6] free
 tau[29,7] free
 tau[30,0] free
 tau[30,1] free
 tau[30,2] free
 tau[30,3] free
 tau[30,4] free
 tau[30,5] free
 tau[30,6] free
 tau[30,7] free
 tau[31,0] free
 tau[31,1] free
 tau[31,2] free
 tau[31,3] free
 tau[31,4] free
 tau[31,5] free
 tau[31,6] free
 tau[31,7] free
 tau[32,0] free
 tau[32,1] free
 tau[32,2] free
 tau[32,3] free
 tau[32,4] free
 tau[32,5] free
 tau[32,6] free
 tau[32,7] free
 tau[33,0] free
 tau[33,1] free
 tau[33,2] free
 tau[33,3] free
 tau[33,4] free
 tau[33,5] free
 tau[33,6] free
 tau[33,7] free
 tau[34,0] free
 tau[34,1] free
 tau[34,2] free
 tau[34,3] free
 tau[34,4] free
 tau[34,5] free
 tau[34,6] free
 tau[34,7] free
 tau[35,0] free
 tau[35,1] free
 tau[35,2] free
 tau[35,3] free
 tau[35,4] free
 tau[35,5] free
 tau[35,6] free
 tau[35,7] free
 tau[36,0] free
 tau[36,1] free
 tau[36,2] free
 tau[36,3] free
 tau[36,4] free
 tau[36,5] free
 tau[36,6] free
 tau[36,7] free
 tau[37,0] free
 tau[37,1] free
 tau[37,2] free
 tau[37,3] free
 tau[37,4] free
 tau[37,5] free
 tau[37,6] free
 tau[37,7] free
 tau[38,0] free
 tau[38,1] free
 tau[38,2] free
 tau[38,3] free
 tau[38,4] free
 tau[38,5] free
 tau[38,6] free
 tau[38,7] free
 tau[39,0] free
 tau[39,1] free
 tau[39,2] free
 tau[39,3] free
 tau[39,4] free
 tau[39,5] free
 tau[39,6] free
 tau[39,7] free
 tau[40,0] free
 tau[40,1] free
 tau[40,2] free
 tau[40,3] free
 tau[40,4] free
 tau[40,5] free
 tau[40,6] free
 tau[40,7] free
 tau[41,0] free
 tau[41,1] free
 tau[41,2] free
 tau[41,3] free
 tau[41,4] free
 tau[41,5] free
 tau[41,6] free
 tau[41,7] free
 tau[42,0] free
 tau[42,1] free
 tau[42,2] free
 tau[42,3] free
 tau[42,4] free
 tau[42,5] free
 tau[42,6] free
 tau[42,7] free
 tau[43,0] free
 tau[43,1] free
 tau[43,2] free
 tau[43,3] free
 tau[43,4] free
 tau[43,5] free
 tau[43,6] free
 tau[43,7] free
 tau[44,0] free
 tau[44,1] free
 tau[44,2] free
 tau[44,3] free
 tau[44,4] free
 tau[44,5] free
 tau[44,6] free
 tau[44,7] free
 tau[45,0] free
 tau[45,1] free
 tau[45,2] free
 tau[45,3] free
 tau[45,4] free
 tau[45,5] free
 tau[45,6] free
 tau[45,7] free
 tau[46,0] free
 tau[46,1] free
 tau[46,2] free
 tau[46,3] free
 tau[46,4] free
 tau[46,5] free
 tau[46,6] free
 tau[46,7] free
 tau[47,0] free
 tau[47,1] free
 tau[47,2] free
 tau[47,3] free
 tau[47,4] free
 tau[47,5] free
 tau[47,6] free
 tau[47,7] free
 tau[48,0] free
 tau[48,1] free
 tau[48,2] free
 tau[48,3] free
 tau[48,4] free
 tau[48,5] free
 tau[48,6] free
 tau[48,7] free
 tau[49,0] free
 tau[49,1] free
 tau[49,2] free
 tau[49,3] free
 tau[49,4] free
 tau[49,5] free
 tau[49,6] free
 tau[49,7] free
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
 r[25,0] <= 1
 r[25,1] <= 1
 r[25,2] <= 1
 r[25,3] <= 1
 r[25,4] <= 1
 r[25,5] <= 1
 r[25,6] <= 1
 r[25,7] <= 1
 r[26,0] <= 1
 r[26,1] <= 1
 r[26,2] <= 1
 r[26,3] <= 1
 r[26,4] <= 1
 r[26,5] <= 1
 r[26,6] <= 1
 r[26,7] <= 1
 r[27,0] <= 1
 r[27,1] <= 1
 r[27,2] <= 1
 r[27,3] <= 1
 r[27,4] <= 1
 r[27,5] <= 1
 r[27,6] <= 1
 r[27,7] <= 1
 r[28,0] <= 1
 r[28,1] <= 1
 r[28,2] <= 1
 r[28,3] <= 1
 r[28,4] <= 1
 r[28,5] <= 1
 r[28,6] <= 1
 r[28,7] <= 1
 r[29,0] <= 1
 r[29,1] <= 1
 r[29,2] <= 1
 r[29,3] <= 1
 r[29,4] <= 1
 r[29,5] <= 1
 r[29,6] <= 1
 r[29,7] <= 1
 r[30,0] <= 1
 r[30,1] <= 1
 r[30,2] <= 1
 r[30,3] <= 1
 r[30,4] <= 1
 r[30,5] <= 1
 r[30,6] <= 1
 r[30,7] <= 1
 r[31,0] <= 1
 r[31,1] <= 1
 r[31,2] <= 1
 r[31,3] <= 1
 r[31,4] <= 1
 r[31,5] <= 1
 r[31,6] <= 1
 r[31,7] <= 1
 r[32,0] <= 1
 r[32,1] <= 1
 r[32,2] <= 1
 r[32,3] <= 1
 r[32,4] <= 1
 r[32,5] <= 1
 r[32,6] <= 1
 r[32,7] <= 1
 r[33,0] <= 1
 r[33,1] <= 1
 r[33,2] <= 1
 r[33,3] <= 1
 r[33,4] <= 1
 r[33,5] <= 1
 r[33,6] <= 1
 r[33,7] <= 1
 r[34,0] <= 1
 r[34,1] <= 1
 r[34,2] <= 1
 r[34,3] <= 1
 r[34,4] <= 1
 r[34,5] <= 1
 r[34,6] <= 1
 r[34,7] <= 1
 r[35,0] <= 1
 r[35,1] <= 1
 r[35,2] <= 1
 r[35,3] <= 1
 r[35,4] <= 1
 r[35,5] <= 1
 r[35,6] <= 1
 r[35,7] <= 1
 r[36,0] <= 1
 r[36,1] <= 1
 r[36,2] <= 1
 r[36,3] <= 1
 r[36,4] <= 1
 r[36,5] <= 1
 r[36,6] <= 1
 r[36,7] <= 1
 r[37,0] <= 1
 r[37,1] <= 1
 r[37,2] <= 1
 r[37,3] <= 1
 r[37,4] <= 1
 r[37,5] <= 1
 r[37,6] <= 1
 r[37,7] <= 1
 r[38,0] <= 1
 r[38,1] <= 1
 r[38,2] <= 1
 r[38,3] <= 1
 r[38,4] <= 1
 r[38,5] <= 1
 r[38,6] <= 1
 r[38,7] <= 1
 r[39,0] <= 1
 r[39,1] <= 1
 r[39,2] <= 1
 r[39,3] <= 1
 r[39,4] <= 1
 r[39,5] <= 1
 r[39,6] <= 1
 r[39,7] <= 1
 r[40,0] <= 1
 r[40,1] <= 1
 r[40,2] <= 1
 r[40,3] <= 1
 r[40,4] <= 1
 r[40,5] <= 1
 r[40,6] <= 1
 r[40,7] <= 1
 r[41,0] <= 1
 r[41,1] <= 1
 r[41,2] <= 1
 r[41,3] <= 1
 r[41,4] <= 1
 r[41,5] <= 1
 r[41,6] <= 1
 r[41,7] <= 1
 r[42,0] <= 1
 r[42,1] <= 1
 r[42,2] <= 1
 r[42,3] <= 1
 r[42,4] <= 1
 r[42,5] <= 1
 r[42,6] <= 1
 r[42,7] <= 1
 r[43,0] <= 1
 r[43,1] <= 1
 r[43,2] <= 1
 r[43,3] <= 1
 r[43,4] <= 1
 r[43,5] <= 1
 r[43,6] <= 1
 r[43,7] <= 1
 r[44,0] <= 1
 r[44,1] <= 1
 r[44,2] <= 1
 r[44,3] <= 1
 r[44,4] <= 1
 r[44,5] <= 1
 r[44,6] <= 1
 r[44,7] <= 1
 r[45,0] <= 1
 r[45,1] <= 1
 r[45,2] <= 1
 r[45,3] <= 1
 r[45,4] <= 1
 r[45,5] <= 1
 r[45,6] <= 1
 r[45,7] <= 1
 r[46,0] <= 1
 r[46,1] <= 1
 r[46,2] <= 1
 r[46,3] <= 1
 r[46,4] <= 1
 r[46,5] <= 1
 r[46,6] <= 1
 r[46,7] <= 1
 r[47,0] <= 1
 r[47,1] <= 1
 r[47,2] <= 1
 r[47,3] <= 1
 r[47,4] <= 1
 r[47,5] <= 1
 r[47,6] <= 1
 r[47,7] <= 1
 r[48,0] <= 1
 r[48,1] <= 1
 r[48,2] <= 1
 r[48,3] <= 1
 r[48,4] <= 1
 r[48,5] <= 1
 r[48,6] <= 1
 r[48,7] <= 1
 r[49,0] <= 1
 r[49,1] <= 1
 r[49,2] <= 1
 r[49,3] <= 1
 r[49,4] <= 1
 r[49,5] <= 1
 r[49,6] <= 1
 r[49,7] <= 1
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
 Q_hat_adjusted[25] free
 Q_hat_adjusted[26] free
 Q_hat_adjusted[27] free
 Q_hat_adjusted[28] free
 Q_hat_adjusted[29] free
 Q_hat_adjusted[30] free
 Q_hat_adjusted[31] free
 Q_hat_adjusted[32] free
 Q_hat_adjusted[33] free
 Q_hat_adjusted[34] free
 Q_hat_adjusted[35] free
 Q_hat_adjusted[36] free
 Q_hat_adjusted[37] free
 Q_hat_adjusted[38] free
 Q_hat_adjusted[39] free
 Q_hat_adjusted[40] free
 Q_hat_adjusted[41] free
 Q_hat_adjusted[42] free
 Q_hat_adjusted[43] free
 Q_hat_adjusted[44] free
 Q_hat_adjusted[45] free
 Q_hat_adjusted[46] free
 Q_hat_adjusted[47] free
 Q_hat_adjusted[48] free
 Q_hat_adjusted[49] free
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
 max_r_helper <= 1
 Left_0_aux_25 free
 Lost_0_aux_25 free
 Left_0_aux_25 free
 Lost_0_aux_25 free
 max_r_helper <= 1
 Left_0_aux_26 free
 Lost_0_aux_26 free
 Left_0_aux_26 free
 Lost_0_aux_26 free
 max_r_helper <= 1
 Left_0_aux_27 free
 Lost_0_aux_27 free
 Left_0_aux_27 free
 Lost_0_aux_27 free
 max_r_helper <= 1
 Left_0_aux_28 free
 Lost_0_aux_28 free
 Left_0_aux_28 free
 Lost_0_aux_28 free
 max_r_helper <= 1
 Left_0_aux_29 free
 Lost_0_aux_29 free
 Left_0_aux_29 free
 Lost_0_aux_29 free
 max_r_helper <= 1
 Left_0_aux_30 free
 Lost_0_aux_30 free
 Left_0_aux_30 free
 Lost_0_aux_30 free
 max_r_helper <= 1
 Left_0_aux_31 free
 Lost_0_aux_31 free
 Left_0_aux_31 free
 Lost_0_aux_31 free
 max_r_helper <= 1
 Left_0_aux_32 free
 Lost_0_aux_32 free
 Left_0_aux_32 free
 Lost_0_aux_32 free
 max_r_helper <= 1
 Left_0_aux_33 free
 Lost_0_aux_33 free
 Left_0_aux_33 free
 Lost_0_aux_33 free
 max_r_helper <= 1
 Left_0_aux_34 free
 Lost_0_aux_34 free
 Left_0_aux_34 free
 Lost_0_aux_34 free
 max_r_helper <= 1
 Left_0_aux_35 free
 Lost_0_aux_35 free
 Left_0_aux_35 free
 Lost_0_aux_35 free
 max_r_helper <= 1
 Left_0_aux_36 free
 Lost_0_aux_36 free
 Left_0_aux_36 free
 Lost_0_aux_36 free
 max_r_helper <= 1
 Left_0_aux_37 free
 Lost_0_aux_37 free
 Left_0_aux_37 free
 Lost_0_aux_37 free
 max_r_helper <= 1
 Left_0_aux_38 free
 Lost_0_aux_38 free
 Left_0_aux_38 free
 Lost_0_aux_38 free
 max_r_helper <= 1
 Left_0_aux_39 free
 Lost_0_aux_39 free
 Left_0_aux_39 free
 Lost_0_aux_39 free
 max_r_helper <= 1
 Left_0_aux_40 free
 Lost_0_aux_40 free
 Left_0_aux_40 free
 Lost_0_aux_40 free
 max_r_helper <= 1
 Left_0_aux_41 free
 Lost_0_aux_41 free
 Left_0_aux_41 free
 Lost_0_aux_41 free
 max_r_helper <= 1
 Left_0_aux_42 free
 Lost_0_aux_42 free
 Left_0_aux_42 free
 Lost_0_aux_42 free
 max_r_helper <= 1
 Left_0_aux_43 free
 Lost_0_aux_43 free
 Left_0_aux_43 free
 Lost_0_aux_43 free
 max_r_helper <= 1
 Left_0_aux_44 free
 Lost_0_aux_44 free
 Left_0_aux_44 free
 Lost_0_aux_44 free
 max_r_helper <= 1
 Left_0_aux_45 free
 Lost_0_aux_45 free
 Left_0_aux_45 free
 Lost_0_aux_45 free
 max_r_helper <= 1
 Left_0_aux_46 free
 Lost_0_aux_46 free
 Left_0_aux_46 free
 Lost_0_aux_46 free
 max_r_helper <= 1
 Left_0_aux_47 free
 Lost_0_aux_47 free
 Left_0_aux_47 free
 Lost_0_aux_47 free
 max_r_helper <= 1
 Left_0_aux_48 free
 Lost_0_aux_48 free
 Left_0_aux_48 free
 Lost_0_aux_48 free
 max_r_helper <= 1
 Left_0_aux_49 free
 Lost_0_aux_49 free
 Left_0_aux_49 free
 Lost_0_aux_49 free
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
 R[24,5] R[24,6] R[24,7] R[25,0] R[25,1] R[25,2] R[25,3] R[25,4] R[25,5]
 R[25,6] R[25,7] R[26,0] R[26,1] R[26,2] R[26,3] R[26,4] R[26,5] R[26,6]
 R[26,7] R[27,0] R[27,1] R[27,2] R[27,3] R[27,4] R[27,5] R[27,6] R[27,7]
 R[28,0] R[28,1] R[28,2] R[28,3] R[28,4] R[28,5] R[28,6] R[28,7] R[29,0]
 R[29,1] R[29,2] R[29,3] R[29,4] R[29,5] R[29,6] R[29,7] R[30,0] R[30,1]
 R[30,2] R[30,3] R[30,4] R[30,5] R[30,6] R[30,7] R[31,0] R[31,1] R[31,2]
 R[31,3] R[31,4] R[31,5] R[31,6] R[31,7] R[32,0] R[32,1] R[32,2] R[32,3]
 R[32,4] R[32,5] R[32,6] R[32,7] R[33,0] R[33,1] R[33,2] R[33,3] R[33,4]
 R[33,5] R[33,6] R[33,7] R[34,0] R[34,1] R[34,2] R[34,3] R[34,4] R[34,5]
 R[34,6] R[34,7] R[35,0] R[35,1] R[35,2] R[35,3] R[35,4] R[35,5] R[35,6]
 R[35,7] R[36,0] R[36,1] R[36,2] R[36,3] R[36,4] R[36,5] R[36,6] R[36,7]
 R[37,0] R[37,1] R[37,2] R[37,3] R[37,4] R[37,5] R[37,6] R[37,7] R[38,0]
 R[38,1] R[38,2] R[38,3] R[38,4] R[38,5] R[38,6] R[38,7] R[39,0] R[39,1]
 R[39,2] R[39,3] R[39,4] R[39,5] R[39,6] R[39,7] R[40,0] R[40,1] R[40,2]
 R[40,3] R[40,4] R[40,5] R[40,6] R[40,7] R[41,0] R[41,1] R[41,2] R[41,3]
 R[41,4] R[41,5] R[41,6] R[41,7] R[42,0] R[42,1] R[42,2] R[42,3] R[42,4]
 R[42,5] R[42,6] R[42,7] R[43,0] R[43,1] R[43,2] R[43,3] R[43,4] R[43,5]
 R[43,6] R[43,7] R[44,0] R[44,1] R[44,2] R[44,3] R[44,4] R[44,5] R[44,6]
 R[44,7] R[45,0] R[45,1] R[45,2] R[45,3] R[45,4] R[45,5] R[45,6] R[45,7]
 R[46,0] R[46,1] R[46,2] R[46,3] R[46,4] R[46,5] R[46,6] R[46,7] R[47,0]
 R[47,1] R[47,2] R[47,3] R[47,4] R[47,5] R[47,6] R[47,7] R[48,0] R[48,1]
 R[48,2] R[48,3] R[48,4] R[48,5] R[48,6] R[48,7] R[49,0] R[49,1] R[49,2]
 R[49,3] R[49,4] R[49,5] R[49,6] R[49,7]
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
 logistic_constraint_25: Fraction_for_second_order_amount[25] =
   LOGISTIC ( f_var[25] )
 GC626: exp_tau_var_25_0 = EXP ( tau[25,0] )
 GC627: exp_tau_var_25_1 = EXP ( tau[25,1] )
 GC628: exp_tau_var_25_2 = EXP ( tau[25,2] )
 GC629: exp_tau_var_25_3 = EXP ( tau[25,3] )
 GC630: exp_tau_var_25_4 = EXP ( tau[25,4] )
 GC631: exp_tau_var_25_5 = EXP ( tau[25,5] )
 GC632: exp_tau_var_25_6 = EXP ( tau[25,6] )
 GC633: exp_tau_var_25_7 = EXP ( tau[25,7] )
 MaxRConstraint_25: max_r_helper = MAX ( r[25,0] , r[25,1] , r[25,2] ,
   r[25,3] , r[25,4] , r[25,5] , r[25,6] , r[25,7] )
 Max_Constraint_25: Q1_var[25] = MAX ( Q_hat_adjusted[25] , 0 )
 Constr_Sold_0_min_25: Sold_0[25] = MIN (
   Total_Demand_Up_to_K_minus_1[25] , Q0_var[25] )
 Constr_Left_0_max_25: Left_0[25] = MAX ( Left_0_aux_25 , 0 )
 Constr_Lost_0_max_25: Lost_0[25] = MAX ( Lost_0_aux_25 , 0 )
 Constr_Sold_1_min_25: Sold_1[25] = MIN ( Total_Demand_from_k_to_T[25] ,
   Q1_plus_left[25] )
 Constr_Left_1_max_25: Left_1[25] = MAX ( Left_0_aux_25 , 0 )
 Constr_Lost_1_max_25: Lost_1[25] = MAX ( Lost_0_aux_25 , 0 )
 logistic_constraint_26: Fraction_for_second_order_amount[26] =
   LOGISTIC ( f_var[26] )
 GC651: exp_tau_var_26_0 = EXP ( tau[26,0] )
 GC652: exp_tau_var_26_1 = EXP ( tau[26,1] )
 GC653: exp_tau_var_26_2 = EXP ( tau[26,2] )
 GC654: exp_tau_var_26_3 = EXP ( tau[26,3] )
 GC655: exp_tau_var_26_4 = EXP ( tau[26,4] )
 GC656: exp_tau_var_26_5 = EXP ( tau[26,5] )
 GC657: exp_tau_var_26_6 = EXP ( tau[26,6] )
 GC658: exp_tau_var_26_7 = EXP ( tau[26,7] )
 MaxRConstraint_26: max_r_helper = MAX ( r[26,0] , r[26,1] , r[26,2] ,
   r[26,3] , r[26,4] , r[26,5] , r[26,6] , r[26,7] )
 Max_Constraint_26: Q1_var[26] = MAX ( Q_hat_adjusted[26] , 0 )
 Constr_Sold_0_min_26: Sold_0[26] = MIN (
   Total_Demand_Up_to_K_minus_1[26] , Q0_var[26] )
 Constr_Left_0_max_26: Left_0[26] = MAX ( Left_0_aux_26 , 0 )
 Constr_Lost_0_max_26: Lost_0[26] = MAX ( Lost_0_aux_26 , 0 )
 Constr_Sold_1_min_26: Sold_1[26] = MIN ( Total_Demand_from_k_to_T[26] ,
   Q1_plus_left[26] )
 Constr_Left_1_max_26: Left_1[26] = MAX ( Left_0_aux_26 , 0 )
 Constr_Lost_1_max_26: Lost_1[26] = MAX ( Lost_0_aux_26 , 0 )
 logistic_constraint_27: Fraction_for_second_order_amount[27] =
   LOGISTIC ( f_var[27] )
 GC676: exp_tau_var_27_0 = EXP ( tau[27,0] )
 GC677: exp_tau_var_27_1 = EXP ( tau[27,1] )
 GC678: exp_tau_var_27_2 = EXP ( tau[27,2] )
 GC679: exp_tau_var_27_3 = EXP ( tau[27,3] )
 GC680: exp_tau_var_27_4 = EXP ( tau[27,4] )
 GC681: exp_tau_var_27_5 = EXP ( tau[27,5] )
 GC682: exp_tau_var_27_6 = EXP ( tau[27,6] )
 GC683: exp_tau_var_27_7 = EXP ( tau[27,7] )
 MaxRConstraint_27: max_r_helper = MAX ( r[27,0] , r[27,1] , r[27,2] ,
   r[27,3] , r[27,4] , r[27,5] , r[27,6] , r[27,7] )
 Max_Constraint_27: Q1_var[27] = MAX ( Q_hat_adjusted[27] , 0 )
 Constr_Sold_0_min_27: Sold_0[27] = MIN (
   Total_Demand_Up_to_K_minus_1[27] , Q0_var[27] )
 Constr_Left_0_max_27: Left_0[27] = MAX ( Left_0_aux_27 , 0 )
 Constr_Lost_0_max_27: Lost_0[27] = MAX ( Lost_0_aux_27 , 0 )
 Constr_Sold_1_min_27: Sold_1[27] = MIN ( Total_Demand_from_k_to_T[27] ,
   Q1_plus_left[27] )
 Constr_Left_1_max_27: Left_1[27] = MAX ( Left_0_aux_27 , 0 )
 Constr_Lost_1_max_27: Lost_1[27] = MAX ( Lost_0_aux_27 , 0 )
 logistic_constraint_28: Fraction_for_second_order_amount[28] =
   LOGISTIC ( f_var[28] )
 GC701: exp_tau_var_28_0 = EXP ( tau[28,0] )
 GC702: exp_tau_var_28_1 = EXP ( tau[28,1] )
 GC703: exp_tau_var_28_2 = EXP ( tau[28,2] )
 GC704: exp_tau_var_28_3 = EXP ( tau[28,3] )
 GC705: exp_tau_var_28_4 = EXP ( tau[28,4] )
 GC706: exp_tau_var_28_5 = EXP ( tau[28,5] )
 GC707: exp_tau_var_28_6 = EXP ( tau[28,6] )
 GC708: exp_tau_var_28_7 = EXP ( tau[28,7] )
 MaxRConstraint_28: max_r_helper = MAX ( r[28,0] , r[28,1] , r[28,2] ,
   r[28,3] , r[28,4] , r[28,5] , r[28,6] , r[28,7] )
 Max_Constraint_28: Q1_var[28] = MAX ( Q_hat_adjusted[28] , 0 )
 Constr_Sold_0_min_28: Sold_0[28] = MIN (
   Total_Demand_Up_to_K_minus_1[28] , Q0_var[28] )
 Constr_Left_0_max_28: Left_0[28] = MAX ( Left_0_aux_28 , 0 )
 Constr_Lost_0_max_28: Lost_0[28] = MAX ( Lost_0_aux_28 , 0 )
 Constr_Sold_1_min_28: Sold_1[28] = MIN ( Total_Demand_from_k_to_T[28] ,
   Q1_plus_left[28] )
 Constr_Left_1_max_28: Left_1[28] = MAX ( Left_0_aux_28 , 0 )
 Constr_Lost_1_max_28: Lost_1[28] = MAX ( Lost_0_aux_28 , 0 )
 logistic_constraint_29: Fraction_for_second_order_amount[29] =
   LOGISTIC ( f_var[29] )
 GC726: exp_tau_var_29_0 = EXP ( tau[29,0] )
 GC727: exp_tau_var_29_1 = EXP ( tau[29,1] )
 GC728: exp_tau_var_29_2 = EXP ( tau[29,2] )
 GC729: exp_tau_var_29_3 = EXP ( tau[29,3] )
 GC730: exp_tau_var_29_4 = EXP ( tau[29,4] )
 GC731: exp_tau_var_29_5 = EXP ( tau[29,5] )
 GC732: exp_tau_var_29_6 = EXP ( tau[29,6] )
 GC733: exp_tau_var_29_7 = EXP ( tau[29,7] )
 MaxRConstraint_29: max_r_helper = MAX ( r[29,0] , r[29,1] , r[29,2] ,
   r[29,3] , r[29,4] , r[29,5] , r[29,6] , r[29,7] )
 Max_Constraint_29: Q1_var[29] = MAX ( Q_hat_adjusted[29] , 0 )
 Constr_Sold_0_min_29: Sold_0[29] = MIN (
   Total_Demand_Up_to_K_minus_1[29] , Q0_var[29] )
 Constr_Left_0_max_29: Left_0[29] = MAX ( Left_0_aux_29 , 0 )
 Constr_Lost_0_max_29: Lost_0[29] = MAX ( Lost_0_aux_29 , 0 )
 Constr_Sold_1_min_29: Sold_1[29] = MIN ( Total_Demand_from_k_to_T[29] ,
   Q1_plus_left[29] )
 Constr_Left_1_max_29: Left_1[29] = MAX ( Left_0_aux_29 , 0 )
 Constr_Lost_1_max_29: Lost_1[29] = MAX ( Lost_0_aux_29 , 0 )
 logistic_constraint_30: Fraction_for_second_order_amount[30] =
   LOGISTIC ( f_var[30] )
 GC751: exp_tau_var_30_0 = EXP ( tau[30,0] )
 GC752: exp_tau_var_30_1 = EXP ( tau[30,1] )
 GC753: exp_tau_var_30_2 = EXP ( tau[30,2] )
 GC754: exp_tau_var_30_3 = EXP ( tau[30,3] )
 GC755: exp_tau_var_30_4 = EXP ( tau[30,4] )
 GC756: exp_tau_var_30_5 = EXP ( tau[30,5] )
 GC757: exp_tau_var_30_6 = EXP ( tau[30,6] )
 GC758: exp_tau_var_30_7 = EXP ( tau[30,7] )
 MaxRConstraint_30: max_r_helper = MAX ( r[30,0] , r[30,1] , r[30,2] ,
   r[30,3] , r[30,4] , r[30,5] , r[30,6] , r[30,7] )
 Max_Constraint_30: Q1_var[30] = MAX ( Q_hat_adjusted[30] , 0 )
 Constr_Sold_0_min_30: Sold_0[30] = MIN (
   Total_Demand_Up_to_K_minus_1[30] , Q0_var[30] )
 Constr_Left_0_max_30: Left_0[30] = MAX ( Left_0_aux_30 , 0 )
 Constr_Lost_0_max_30: Lost_0[30] = MAX ( Lost_0_aux_30 , 0 )
 Constr_Sold_1_min_30: Sold_1[30] = MIN ( Total_Demand_from_k_to_T[30] ,
   Q1_plus_left[30] )
 Constr_Left_1_max_30: Left_1[30] = MAX ( Left_0_aux_30 , 0 )
 Constr_Lost_1_max_30: Lost_1[30] = MAX ( Lost_0_aux_30 , 0 )
 logistic_constraint_31: Fraction_for_second_order_amount[31] =
   LOGISTIC ( f_var[31] )
 GC776: exp_tau_var_31_0 = EXP ( tau[31,0] )
 GC777: exp_tau_var_31_1 = EXP ( tau[31,1] )
 GC778: exp_tau_var_31_2 = EXP ( tau[31,2] )
 GC779: exp_tau_var_31_3 = EXP ( tau[31,3] )
 GC780: exp_tau_var_31_4 = EXP ( tau[31,4] )
 GC781: exp_tau_var_31_5 = EXP ( tau[31,5] )
 GC782: exp_tau_var_31_6 = EXP ( tau[31,6] )
 GC783: exp_tau_var_31_7 = EXP ( tau[31,7] )
 MaxRConstraint_31: max_r_helper = MAX ( r[31,0] , r[31,1] , r[31,2] ,
   r[31,3] , r[31,4] , r[31,5] , r[31,6] , r[31,7] )
 Max_Constraint_31: Q1_var[31] = MAX ( Q_hat_adjusted[31] , 0 )
 Constr_Sold_0_min_31: Sold_0[31] = MIN (
   Total_Demand_Up_to_K_minus_1[31] , Q0_var[31] )
 Constr_Left_0_max_31: Left_0[31] = MAX ( Left_0_aux_31 , 0 )
 Constr_Lost_0_max_31: Lost_0[31] = MAX ( Lost_0_aux_31 , 0 )
 Constr_Sold_1_min_31: Sold_1[31] = MIN ( Total_Demand_from_k_to_T[31] ,
   Q1_plus_left[31] )
 Constr_Left_1_max_31: Left_1[31] = MAX ( Left_0_aux_31 , 0 )
 Constr_Lost_1_max_31: Lost_1[31] = MAX ( Lost_0_aux_31 , 0 )
 logistic_constraint_32: Fraction_for_second_order_amount[32] =
   LOGISTIC ( f_var[32] )
 GC801: exp_tau_var_32_0 = EXP ( tau[32,0] )
 GC802: exp_tau_var_32_1 = EXP ( tau[32,1] )
 GC803: exp_tau_var_32_2 = EXP ( tau[32,2] )
 GC804: exp_tau_var_32_3 = EXP ( tau[32,3] )
 GC805: exp_tau_var_32_4 = EXP ( tau[32,4] )
 GC806: exp_tau_var_32_5 = EXP ( tau[32,5] )
 GC807: exp_tau_var_32_6 = EXP ( tau[32,6] )
 GC808: exp_tau_var_32_7 = EXP ( tau[32,7] )
 MaxRConstraint_32: max_r_helper = MAX ( r[32,0] , r[32,1] , r[32,2] ,
   r[32,3] , r[32,4] , r[32,5] , r[32,6] , r[32,7] )
 Max_Constraint_32: Q1_var[32] = MAX ( Q_hat_adjusted[32] , 0 )
 Constr_Sold_0_min_32: Sold_0[32] = MIN (
   Total_Demand_Up_to_K_minus_1[32] , Q0_var[32] )
 Constr_Left_0_max_32: Left_0[32] = MAX ( Left_0_aux_32 , 0 )
 Constr_Lost_0_max_32: Lost_0[32] = MAX ( Lost_0_aux_32 , 0 )
 Constr_Sold_1_min_32: Sold_1[32] = MIN ( Total_Demand_from_k_to_T[32] ,
   Q1_plus_left[32] )
 Constr_Left_1_max_32: Left_1[32] = MAX ( Left_0_aux_32 , 0 )
 Constr_Lost_1_max_32: Lost_1[32] = MAX ( Lost_0_aux_32 , 0 )
 logistic_constraint_33: Fraction_for_second_order_amount[33] =
   LOGISTIC ( f_var[33] )
 GC826: exp_tau_var_33_0 = EXP ( tau[33,0] )
 GC827: exp_tau_var_33_1 = EXP ( tau[33,1] )
 GC828: exp_tau_var_33_2 = EXP ( tau[33,2] )
 GC829: exp_tau_var_33_3 = EXP ( tau[33,3] )
 GC830: exp_tau_var_33_4 = EXP ( tau[33,4] )
 GC831: exp_tau_var_33_5 = EXP ( tau[33,5] )
 GC832: exp_tau_var_33_6 = EXP ( tau[33,6] )
 GC833: exp_tau_var_33_7 = EXP ( tau[33,7] )
 MaxRConstraint_33: max_r_helper = MAX ( r[33,0] , r[33,1] , r[33,2] ,
   r[33,3] , r[33,4] , r[33,5] , r[33,6] , r[33,7] )
 Max_Constraint_33: Q1_var[33] = MAX ( Q_hat_adjusted[33] , 0 )
 Constr_Sold_0_min_33: Sold_0[33] = MIN (
   Total_Demand_Up_to_K_minus_1[33] , Q0_var[33] )
 Constr_Left_0_max_33: Left_0[33] = MAX ( Left_0_aux_33 , 0 )
 Constr_Lost_0_max_33: Lost_0[33] = MAX ( Lost_0_aux_33 , 0 )
 Constr_Sold_1_min_33: Sold_1[33] = MIN ( Total_Demand_from_k_to_T[33] ,
   Q1_plus_left[33] )
 Constr_Left_1_max_33: Left_1[33] = MAX ( Left_0_aux_33 , 0 )
 Constr_Lost_1_max_33: Lost_1[33] = MAX ( Lost_0_aux_33 , 0 )
 logistic_constraint_34: Fraction_for_second_order_amount[34] =
   LOGISTIC ( f_var[34] )
 GC851: exp_tau_var_34_0 = EXP ( tau[34,0] )
 GC852: exp_tau_var_34_1 = EXP ( tau[34,1] )
 GC853: exp_tau_var_34_2 = EXP ( tau[34,2] )
 GC854: exp_tau_var_34_3 = EXP ( tau[34,3] )
 GC855: exp_tau_var_34_4 = EXP ( tau[34,4] )
 GC856: exp_tau_var_34_5 = EXP ( tau[34,5] )
 GC857: exp_tau_var_34_6 = EXP ( tau[34,6] )
 GC858: exp_tau_var_34_7 = EXP ( tau[34,7] )
 MaxRConstraint_34: max_r_helper = MAX ( r[34,0] , r[34,1] , r[34,2] ,
   r[34,3] , r[34,4] , r[34,5] , r[34,6] , r[34,7] )
 Max_Constraint_34: Q1_var[34] = MAX ( Q_hat_adjusted[34] , 0 )
 Constr_Sold_0_min_34: Sold_0[34] = MIN (
   Total_Demand_Up_to_K_minus_1[34] , Q0_var[34] )
 Constr_Left_0_max_34: Left_0[34] = MAX ( Left_0_aux_34 , 0 )
 Constr_Lost_0_max_34: Lost_0[34] = MAX ( Lost_0_aux_34 , 0 )
 Constr_Sold_1_min_34: Sold_1[34] = MIN ( Total_Demand_from_k_to_T[34] ,
   Q1_plus_left[34] )
 Constr_Left_1_max_34: Left_1[34] = MAX ( Left_0_aux_34 , 0 )
 Constr_Lost_1_max_34: Lost_1[34] = MAX ( Lost_0_aux_34 , 0 )
 logistic_constraint_35: Fraction_for_second_order_amount[35] =
   LOGISTIC ( f_var[35] )
 GC876: exp_tau_var_35_0 = EXP ( tau[35,0] )
 GC877: exp_tau_var_35_1 = EXP ( tau[35,1] )
 GC878: exp_tau_var_35_2 = EXP ( tau[35,2] )
 GC879: exp_tau_var_35_3 = EXP ( tau[35,3] )
 GC880: exp_tau_var_35_4 = EXP ( tau[35,4] )
 GC881: exp_tau_var_35_5 = EXP ( tau[35,5] )
 GC882: exp_tau_var_35_6 = EXP ( tau[35,6] )
 GC883: exp_tau_var_35_7 = EXP ( tau[35,7] )
 MaxRConstraint_35: max_r_helper = MAX ( r[35,0] , r[35,1] , r[35,2] ,
   r[35,3] , r[35,4] , r[35,5] , r[35,6] , r[35,7] )
 Max_Constraint_35: Q1_var[35] = MAX ( Q_hat_adjusted[35] , 0 )
 Constr_Sold_0_min_35: Sold_0[35] = MIN (
   Total_Demand_Up_to_K_minus_1[35] , Q0_var[35] )
 Constr_Left_0_max_35: Left_0[35] = MAX ( Left_0_aux_35 , 0 )
 Constr_Lost_0_max_35: Lost_0[35] = MAX ( Lost_0_aux_35 , 0 )
 Constr_Sold_1_min_35: Sold_1[35] = MIN ( Total_Demand_from_k_to_T[35] ,
   Q1_plus_left[35] )
 Constr_Left_1_max_35: Left_1[35] = MAX ( Left_0_aux_35 , 0 )
 Constr_Lost_1_max_35: Lost_1[35] = MAX ( Lost_0_aux_35 , 0 )
 logistic_constraint_36: Fraction_for_second_order_amount[36] =
   LOGISTIC ( f_var[36] )
 GC901: exp_tau_var_36_0 = EXP ( tau[36,0] )
 GC902: exp_tau_var_36_1 = EXP ( tau[36,1] )
 GC903: exp_tau_var_36_2 = EXP ( tau[36,2] )
 GC904: exp_tau_var_36_3 = EXP ( tau[36,3] )
 GC905: exp_tau_var_36_4 = EXP ( tau[36,4] )
 GC906: exp_tau_var_36_5 = EXP ( tau[36,5] )
 GC907: exp_tau_var_36_6 = EXP ( tau[36,6] )
 GC908: exp_tau_var_36_7 = EXP ( tau[36,7] )
 MaxRConstraint_36: max_r_helper = MAX ( r[36,0] , r[36,1] , r[36,2] ,
   r[36,3] , r[36,4] , r[36,5] , r[36,6] , r[36,7] )
 Max_Constraint_36: Q1_var[36] = MAX ( Q_hat_adjusted[36] , 0 )
 Constr_Sold_0_min_36: Sold_0[36] = MIN (
   Total_Demand_Up_to_K_minus_1[36] , Q0_var[36] )
 Constr_Left_0_max_36: Left_0[36] = MAX ( Left_0_aux_36 , 0 )
 Constr_Lost_0_max_36: Lost_0[36] = MAX ( Lost_0_aux_36 , 0 )
 Constr_Sold_1_min_36: Sold_1[36] = MIN ( Total_Demand_from_k_to_T[36] ,
   Q1_plus_left[36] )
 Constr_Left_1_max_36: Left_1[36] = MAX ( Left_0_aux_36 , 0 )
 Constr_Lost_1_max_36: Lost_1[36] = MAX ( Lost_0_aux_36 , 0 )
 logistic_constraint_37: Fraction_for_second_order_amount[37] =
   LOGISTIC ( f_var[37] )
 GC926: exp_tau_var_37_0 = EXP ( tau[37,0] )
 GC927: exp_tau_var_37_1 = EXP ( tau[37,1] )
 GC928: exp_tau_var_37_2 = EXP ( tau[37,2] )
 GC929: exp_tau_var_37_3 = EXP ( tau[37,3] )
 GC930: exp_tau_var_37_4 = EXP ( tau[37,4] )
 GC931: exp_tau_var_37_5 = EXP ( tau[37,5] )
 GC932: exp_tau_var_37_6 = EXP ( tau[37,6] )
 GC933: exp_tau_var_37_7 = EXP ( tau[37,7] )
 MaxRConstraint_37: max_r_helper = MAX ( r[37,0] , r[37,1] , r[37,2] ,
   r[37,3] , r[37,4] , r[37,5] , r[37,6] , r[37,7] )
 Max_Constraint_37: Q1_var[37] = MAX ( Q_hat_adjusted[37] , 0 )
 Constr_Sold_0_min_37: Sold_0[37] = MIN (
   Total_Demand_Up_to_K_minus_1[37] , Q0_var[37] )
 Constr_Left_0_max_37: Left_0[37] = MAX ( Left_0_aux_37 , 0 )
 Constr_Lost_0_max_37: Lost_0[37] = MAX ( Lost_0_aux_37 , 0 )
 Constr_Sold_1_min_37: Sold_1[37] = MIN ( Total_Demand_from_k_to_T[37] ,
   Q1_plus_left[37] )
 Constr_Left_1_max_37: Left_1[37] = MAX ( Left_0_aux_37 , 0 )
 Constr_Lost_1_max_37: Lost_1[37] = MAX ( Lost_0_aux_37 , 0 )
 logistic_constraint_38: Fraction_for_second_order_amount[38] =
   LOGISTIC ( f_var[38] )
 GC951: exp_tau_var_38_0 = EXP ( tau[38,0] )
 GC952: exp_tau_var_38_1 = EXP ( tau[38,1] )
 GC953: exp_tau_var_38_2 = EXP ( tau[38,2] )
 GC954: exp_tau_var_38_3 = EXP ( tau[38,3] )
 GC955: exp_tau_var_38_4 = EXP ( tau[38,4] )
 GC956: exp_tau_var_38_5 = EXP ( tau[38,5] )
 GC957: exp_tau_var_38_6 = EXP ( tau[38,6] )
 GC958: exp_tau_var_38_7 = EXP ( tau[38,7] )
 MaxRConstraint_38: max_r_helper = MAX ( r[38,0] , r[38,1] , r[38,2] ,
   r[38,3] , r[38,4] , r[38,5] , r[38,6] , r[38,7] )
 Max_Constraint_38: Q1_var[38] = MAX ( Q_hat_adjusted[38] , 0 )
 Constr_Sold_0_min_38: Sold_0[38] = MIN (
   Total_Demand_Up_to_K_minus_1[38] , Q0_var[38] )
 Constr_Left_0_max_38: Left_0[38] = MAX ( Left_0_aux_38 , 0 )
 Constr_Lost_0_max_38: Lost_0[38] = MAX ( Lost_0_aux_38 , 0 )
 Constr_Sold_1_min_38: Sold_1[38] = MIN ( Total_Demand_from_k_to_T[38] ,
   Q1_plus_left[38] )
 Constr_Left_1_max_38: Left_1[38] = MAX ( Left_0_aux_38 , 0 )
 Constr_Lost_1_max_38: Lost_1[38] = MAX ( Lost_0_aux_38 , 0 )
 logistic_constraint_39: Fraction_for_second_order_amount[39] =
   LOGISTIC ( f_var[39] )
 GC976: exp_tau_var_39_0 = EXP ( tau[39,0] )
 GC977: exp_tau_var_39_1 = EXP ( tau[39,1] )
 GC978: exp_tau_var_39_2 = EXP ( tau[39,2] )
 GC979: exp_tau_var_39_3 = EXP ( tau[39,3] )
 GC980: exp_tau_var_39_4 = EXP ( tau[39,4] )
 GC981: exp_tau_var_39_5 = EXP ( tau[39,5] )
 GC982: exp_tau_var_39_6 = EXP ( tau[39,6] )
 GC983: exp_tau_var_39_7 = EXP ( tau[39,7] )
 MaxRConstraint_39: max_r_helper = MAX ( r[39,0] , r[39,1] , r[39,2] ,
   r[39,3] , r[39,4] , r[39,5] , r[39,6] , r[39,7] )
 Max_Constraint_39: Q1_var[39] = MAX ( Q_hat_adjusted[39] , 0 )
 Constr_Sold_0_min_39: Sold_0[39] = MIN (
   Total_Demand_Up_to_K_minus_1[39] , Q0_var[39] )
 Constr_Left_0_max_39: Left_0[39] = MAX ( Left_0_aux_39 , 0 )
 Constr_Lost_0_max_39: Lost_0[39] = MAX ( Lost_0_aux_39 , 0 )
 Constr_Sold_1_min_39: Sold_1[39] = MIN ( Total_Demand_from_k_to_T[39] ,
   Q1_plus_left[39] )
 Constr_Left_1_max_39: Left_1[39] = MAX ( Left_0_aux_39 , 0 )
 Constr_Lost_1_max_39: Lost_1[39] = MAX ( Lost_0_aux_39 , 0 )
 logistic_constraint_40: Fraction_for_second_order_amount[40] =
   LOGISTIC ( f_var[40] )
 GC1001: exp_tau_var_40_0 = EXP ( tau[40,0] )
 GC1002: exp_tau_var_40_1 = EXP ( tau[40,1] )
 GC1003: exp_tau_var_40_2 = EXP ( tau[40,2] )
 GC1004: exp_tau_var_40_3 = EXP ( tau[40,3] )
 GC1005: exp_tau_var_40_4 = EXP ( tau[40,4] )
 GC1006: exp_tau_var_40_5 = EXP ( tau[40,5] )
 GC1007: exp_tau_var_40_6 = EXP ( tau[40,6] )
 GC1008: exp_tau_var_40_7 = EXP ( tau[40,7] )
 MaxRConstraint_40: max_r_helper = MAX ( r[40,0] , r[40,1] , r[40,2] ,
   r[40,3] , r[40,4] , r[40,5] , r[40,6] , r[40,7] )
 Max_Constraint_40: Q1_var[40] = MAX ( Q_hat_adjusted[40] , 0 )
 Constr_Sold_0_min_40: Sold_0[40] = MIN (
   Total_Demand_Up_to_K_minus_1[40] , Q0_var[40] )
 Constr_Left_0_max_40: Left_0[40] = MAX ( Left_0_aux_40 , 0 )
 Constr_Lost_0_max_40: Lost_0[40] = MAX ( Lost_0_aux_40 , 0 )
 Constr_Sold_1_min_40: Sold_1[40] = MIN ( Total_Demand_from_k_to_T[40] ,
   Q1_plus_left[40] )
 Constr_Left_1_max_40: Left_1[40] = MAX ( Left_0_aux_40 , 0 )
 Constr_Lost_1_max_40: Lost_1[40] = MAX ( Lost_0_aux_40 , 0 )
 logistic_constraint_41: Fraction_for_second_order_amount[41] =
   LOGISTIC ( f_var[41] )
 GC1026: exp_tau_var_41_0 = EXP ( tau[41,0] )
 GC1027: exp_tau_var_41_1 = EXP ( tau[41,1] )
 GC1028: exp_tau_var_41_2 = EXP ( tau[41,2] )
 GC1029: exp_tau_var_41_3 = EXP ( tau[41,3] )
 GC1030: exp_tau_var_41_4 = EXP ( tau[41,4] )
 GC1031: exp_tau_var_41_5 = EXP ( tau[41,5] )
 GC1032: exp_tau_var_41_6 = EXP ( tau[41,6] )
 GC1033: exp_tau_var_41_7 = EXP ( tau[41,7] )
 MaxRConstraint_41: max_r_helper = MAX ( r[41,0] , r[41,1] , r[41,2] ,
   r[41,3] , r[41,4] , r[41,5] , r[41,6] , r[41,7] )
 Max_Constraint_41: Q1_var[41] = MAX ( Q_hat_adjusted[41] , 0 )
 Constr_Sold_0_min_41: Sold_0[41] = MIN (
   Total_Demand_Up_to_K_minus_1[41] , Q0_var[41] )
 Constr_Left_0_max_41: Left_0[41] = MAX ( Left_0_aux_41 , 0 )
 Constr_Lost_0_max_41: Lost_0[41] = MAX ( Lost_0_aux_41 , 0 )
 Constr_Sold_1_min_41: Sold_1[41] = MIN ( Total_Demand_from_k_to_T[41] ,
   Q1_plus_left[41] )
 Constr_Left_1_max_41: Left_1[41] = MAX ( Left_0_aux_41 , 0 )
 Constr_Lost_1_max_41: Lost_1[41] = MAX ( Lost_0_aux_41 , 0 )
 logistic_constraint_42: Fraction_for_second_order_amount[42] =
   LOGISTIC ( f_var[42] )
 GC1051: exp_tau_var_42_0 = EXP ( tau[42,0] )
 GC1052: exp_tau_var_42_1 = EXP ( tau[42,1] )
 GC1053: exp_tau_var_42_2 = EXP ( tau[42,2] )
 GC1054: exp_tau_var_42_3 = EXP ( tau[42,3] )
 GC1055: exp_tau_var_42_4 = EXP ( tau[42,4] )
 GC1056: exp_tau_var_42_5 = EXP ( tau[42,5] )
 GC1057: exp_tau_var_42_6 = EXP ( tau[42,6] )
 GC1058: exp_tau_var_42_7 = EXP ( tau[42,7] )
 MaxRConstraint_42: max_r_helper = MAX ( r[42,0] , r[42,1] , r[42,2] ,
   r[42,3] , r[42,4] , r[42,5] , r[42,6] , r[42,7] )
 Max_Constraint_42: Q1_var[42] = MAX ( Q_hat_adjusted[42] , 0 )
 Constr_Sold_0_min_42: Sold_0[42] = MIN (
   Total_Demand_Up_to_K_minus_1[42] , Q0_var[42] )
 Constr_Left_0_max_42: Left_0[42] = MAX ( Left_0_aux_42 , 0 )
 Constr_Lost_0_max_42: Lost_0[42] = MAX ( Lost_0_aux_42 , 0 )
 Constr_Sold_1_min_42: Sold_1[42] = MIN ( Total_Demand_from_k_to_T[42] ,
   Q1_plus_left[42] )
 Constr_Left_1_max_42: Left_1[42] = MAX ( Left_0_aux_42 , 0 )
 Constr_Lost_1_max_42: Lost_1[42] = MAX ( Lost_0_aux_42 , 0 )
 logistic_constraint_43: Fraction_for_second_order_amount[43] =
   LOGISTIC ( f_var[43] )
 GC1076: exp_tau_var_43_0 = EXP ( tau[43,0] )
 GC1077: exp_tau_var_43_1 = EXP ( tau[43,1] )
 GC1078: exp_tau_var_43_2 = EXP ( tau[43,2] )
 GC1079: exp_tau_var_43_3 = EXP ( tau[43,3] )
 GC1080: exp_tau_var_43_4 = EXP ( tau[43,4] )
 GC1081: exp_tau_var_43_5 = EXP ( tau[43,5] )
 GC1082: exp_tau_var_43_6 = EXP ( tau[43,6] )
 GC1083: exp_tau_var_43_7 = EXP ( tau[43,7] )
 MaxRConstraint_43: max_r_helper = MAX ( r[43,0] , r[43,1] , r[43,2] ,
   r[43,3] , r[43,4] , r[43,5] , r[43,6] , r[43,7] )
 Max_Constraint_43: Q1_var[43] = MAX ( Q_hat_adjusted[43] , 0 )
 Constr_Sold_0_min_43: Sold_0[43] = MIN (
   Total_Demand_Up_to_K_minus_1[43] , Q0_var[43] )
 Constr_Left_0_max_43: Left_0[43] = MAX ( Left_0_aux_43 , 0 )
 Constr_Lost_0_max_43: Lost_0[43] = MAX ( Lost_0_aux_43 , 0 )
 Constr_Sold_1_min_43: Sold_1[43] = MIN ( Total_Demand_from_k_to_T[43] ,
   Q1_plus_left[43] )
 Constr_Left_1_max_43: Left_1[43] = MAX ( Left_0_aux_43 , 0 )
 Constr_Lost_1_max_43: Lost_1[43] = MAX ( Lost_0_aux_43 , 0 )
 logistic_constraint_44: Fraction_for_second_order_amount[44] =
   LOGISTIC ( f_var[44] )
 GC1101: exp_tau_var_44_0 = EXP ( tau[44,0] )
 GC1102: exp_tau_var_44_1 = EXP ( tau[44,1] )
 GC1103: exp_tau_var_44_2 = EXP ( tau[44,2] )
 GC1104: exp_tau_var_44_3 = EXP ( tau[44,3] )
 GC1105: exp_tau_var_44_4 = EXP ( tau[44,4] )
 GC1106: exp_tau_var_44_5 = EXP ( tau[44,5] )
 GC1107: exp_tau_var_44_6 = EXP ( tau[44,6] )
 GC1108: exp_tau_var_44_7 = EXP ( tau[44,7] )
 MaxRConstraint_44: max_r_helper = MAX ( r[44,0] , r[44,1] , r[44,2] ,
   r[44,3] , r[44,4] , r[44,5] , r[44,6] , r[44,7] )
 Max_Constraint_44: Q1_var[44] = MAX ( Q_hat_adjusted[44] , 0 )
 Constr_Sold_0_min_44: Sold_0[44] = MIN (
   Total_Demand_Up_to_K_minus_1[44] , Q0_var[44] )
 Constr_Left_0_max_44: Left_0[44] = MAX ( Left_0_aux_44 , 0 )
 Constr_Lost_0_max_44: Lost_0[44] = MAX ( Lost_0_aux_44 , 0 )
 Constr_Sold_1_min_44: Sold_1[44] = MIN ( Total_Demand_from_k_to_T[44] ,
   Q1_plus_left[44] )
 Constr_Left_1_max_44: Left_1[44] = MAX ( Left_0_aux_44 , 0 )
 Constr_Lost_1_max_44: Lost_1[44] = MAX ( Lost_0_aux_44 , 0 )
 logistic_constraint_45: Fraction_for_second_order_amount[45] =
   LOGISTIC ( f_var[45] )
 GC1126: exp_tau_var_45_0 = EXP ( tau[45,0] )
 GC1127: exp_tau_var_45_1 = EXP ( tau[45,1] )
 GC1128: exp_tau_var_45_2 = EXP ( tau[45,2] )
 GC1129: exp_tau_var_45_3 = EXP ( tau[45,3] )
 GC1130: exp_tau_var_45_4 = EXP ( tau[45,4] )
 GC1131: exp_tau_var_45_5 = EXP ( tau[45,5] )
 GC1132: exp_tau_var_45_6 = EXP ( tau[45,6] )
 GC1133: exp_tau_var_45_7 = EXP ( tau[45,7] )
 MaxRConstraint_45: max_r_helper = MAX ( r[45,0] , r[45,1] , r[45,2] ,
   r[45,3] , r[45,4] , r[45,5] , r[45,6] , r[45,7] )
 Max_Constraint_45: Q1_var[45] = MAX ( Q_hat_adjusted[45] , 0 )
 Constr_Sold_0_min_45: Sold_0[45] = MIN (
   Total_Demand_Up_to_K_minus_1[45] , Q0_var[45] )
 Constr_Left_0_max_45: Left_0[45] = MAX ( Left_0_aux_45 , 0 )
 Constr_Lost_0_max_45: Lost_0[45] = MAX ( Lost_0_aux_45 , 0 )
 Constr_Sold_1_min_45: Sold_1[45] = MIN ( Total_Demand_from_k_to_T[45] ,
   Q1_plus_left[45] )
 Constr_Left_1_max_45: Left_1[45] = MAX ( Left_0_aux_45 , 0 )
 Constr_Lost_1_max_45: Lost_1[45] = MAX ( Lost_0_aux_45 , 0 )
 logistic_constraint_46: Fraction_for_second_order_amount[46] =
   LOGISTIC ( f_var[46] )
 GC1151: exp_tau_var_46_0 = EXP ( tau[46,0] )
 GC1152: exp_tau_var_46_1 = EXP ( tau[46,1] )
 GC1153: exp_tau_var_46_2 = EXP ( tau[46,2] )
 GC1154: exp_tau_var_46_3 = EXP ( tau[46,3] )
 GC1155: exp_tau_var_46_4 = EXP ( tau[46,4] )
 GC1156: exp_tau_var_46_5 = EXP ( tau[46,5] )
 GC1157: exp_tau_var_46_6 = EXP ( tau[46,6] )
 GC1158: exp_tau_var_46_7 = EXP ( tau[46,7] )
 MaxRConstraint_46: max_r_helper = MAX ( r[46,0] , r[46,1] , r[46,2] ,
   r[46,3] , r[46,4] , r[46,5] , r[46,6] , r[46,7] )
 Max_Constraint_46: Q1_var[46] = MAX ( Q_hat_adjusted[46] , 0 )
 Constr_Sold_0_min_46: Sold_0[46] = MIN (
   Total_Demand_Up_to_K_minus_1[46] , Q0_var[46] )
 Constr_Left_0_max_46: Left_0[46] = MAX ( Left_0_aux_46 , 0 )
 Constr_Lost_0_max_46: Lost_0[46] = MAX ( Lost_0_aux_46 , 0 )
 Constr_Sold_1_min_46: Sold_1[46] = MIN ( Total_Demand_from_k_to_T[46] ,
   Q1_plus_left[46] )
 Constr_Left_1_max_46: Left_1[46] = MAX ( Left_0_aux_46 , 0 )
 Constr_Lost_1_max_46: Lost_1[46] = MAX ( Lost_0_aux_46 , 0 )
 logistic_constraint_47: Fraction_for_second_order_amount[47] =
   LOGISTIC ( f_var[47] )
 GC1176: exp_tau_var_47_0 = EXP ( tau[47,0] )
 GC1177: exp_tau_var_47_1 = EXP ( tau[47,1] )
 GC1178: exp_tau_var_47_2 = EXP ( tau[47,2] )
 GC1179: exp_tau_var_47_3 = EXP ( tau[47,3] )
 GC1180: exp_tau_var_47_4 = EXP ( tau[47,4] )
 GC1181: exp_tau_var_47_5 = EXP ( tau[47,5] )
 GC1182: exp_tau_var_47_6 = EXP ( tau[47,6] )
 GC1183: exp_tau_var_47_7 = EXP ( tau[47,7] )
 MaxRConstraint_47: max_r_helper = MAX ( r[47,0] , r[47,1] , r[47,2] ,
   r[47,3] , r[47,4] , r[47,5] , r[47,6] , r[47,7] )
 Max_Constraint_47: Q1_var[47] = MAX ( Q_hat_adjusted[47] , 0 )
 Constr_Sold_0_min_47: Sold_0[47] = MIN (
   Total_Demand_Up_to_K_minus_1[47] , Q0_var[47] )
 Constr_Left_0_max_47: Left_0[47] = MAX ( Left_0_aux_47 , 0 )
 Constr_Lost_0_max_47: Lost_0[47] = MAX ( Lost_0_aux_47 , 0 )
 Constr_Sold_1_min_47: Sold_1[47] = MIN ( Total_Demand_from_k_to_T[47] ,
   Q1_plus_left[47] )
 Constr_Left_1_max_47: Left_1[47] = MAX ( Left_0_aux_47 , 0 )
 Constr_Lost_1_max_47: Lost_1[47] = MAX ( Lost_0_aux_47 , 0 )
 logistic_constraint_48: Fraction_for_second_order_amount[48] =
   LOGISTIC ( f_var[48] )
 GC1201: exp_tau_var_48_0 = EXP ( tau[48,0] )
 GC1202: exp_tau_var_48_1 = EXP ( tau[48,1] )
 GC1203: exp_tau_var_48_2 = EXP ( tau[48,2] )
 GC1204: exp_tau_var_48_3 = EXP ( tau[48,3] )
 GC1205: exp_tau_var_48_4 = EXP ( tau[48,4] )
 GC1206: exp_tau_var_48_5 = EXP ( tau[48,5] )
 GC1207: exp_tau_var_48_6 = EXP ( tau[48,6] )
 GC1208: exp_tau_var_48_7 = EXP ( tau[48,7] )
 MaxRConstraint_48: max_r_helper = MAX ( r[48,0] , r[48,1] , r[48,2] ,
   r[48,3] , r[48,4] , r[48,5] , r[48,6] , r[48,7] )
 Max_Constraint_48: Q1_var[48] = MAX ( Q_hat_adjusted[48] , 0 )
 Constr_Sold_0_min_48: Sold_0[48] = MIN (
   Total_Demand_Up_to_K_minus_1[48] , Q0_var[48] )
 Constr_Left_0_max_48: Left_0[48] = MAX ( Left_0_aux_48 , 0 )
 Constr_Lost_0_max_48: Lost_0[48] = MAX ( Lost_0_aux_48 , 0 )
 Constr_Sold_1_min_48: Sold_1[48] = MIN ( Total_Demand_from_k_to_T[48] ,
   Q1_plus_left[48] )
 Constr_Left_1_max_48: Left_1[48] = MAX ( Left_0_aux_48 , 0 )
 Constr_Lost_1_max_48: Lost_1[48] = MAX ( Lost_0_aux_48 , 0 )
 logistic_constraint_49: Fraction_for_second_order_amount[49] =
   LOGISTIC ( f_var[49] )
 GC1226: exp_tau_var_49_0 = EXP ( tau[49,0] )
 GC1227: exp_tau_var_49_1 = EXP ( tau[49,1] )
 GC1228: exp_tau_var_49_2 = EXP ( tau[49,2] )
 GC1229: exp_tau_var_49_3 = EXP ( tau[49,3] )
 GC1230: exp_tau_var_49_4 = EXP ( tau[49,4] )
 GC1231: exp_tau_var_49_5 = EXP ( tau[49,5] )
 GC1232: exp_tau_var_49_6 = EXP ( tau[49,6] )
 GC1233: exp_tau_var_49_7 = EXP ( tau[49,7] )
 MaxRConstraint_49: max_r_helper = MAX ( r[49,0] , r[49,1] , r[49,2] ,
   r[49,3] , r[49,4] , r[49,5] , r[49,6] , r[49,7] )
 Max_Constraint_49: Q1_var[49] = MAX ( Q_hat_adjusted[49] , 0 )
 Constr_Sold_0_min_49: Sold_0[49] = MIN (
   Total_Demand_Up_to_K_minus_1[49] , Q0_var[49] )
 Constr_Left_0_max_49: Left_0[49] = MAX ( Left_0_aux_49 , 0 )
 Constr_Lost_0_max_49: Lost_0[49] = MAX ( Lost_0_aux_49 , 0 )
 Constr_Sold_1_min_49: Sold_1[49] = MIN ( Total_Demand_from_k_to_T[49] ,
   Q1_plus_left[49] )
 Constr_Left_1_max_49: Left_1[49] = MAX ( Left_0_aux_49 , 0 )
 Constr_Lost_1_max_49: Lost_1[49] = MAX ( Lost_0_aux_49 , 0 )
End
