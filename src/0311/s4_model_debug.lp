\ Model profit_maximization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0 Holding_Cost_0[0] + 0 Holding_Cost_0[1] + 0 Holding_Cost_0[2]
   + 0 Holding_Cost_0[3] + 0 Holding_Cost_0[4] + 0 Holding_Cost_1[0]
   + 0 Holding_Cost_1[1] + 0 Holding_Cost_1[2] + 0 Holding_Cost_1[3]
   + 0 Holding_Cost_1[4] + profits_vars[0] + profits_vars[1]
   + profits_vars[2] + profits_vars[3] + profits_vars[4]
Subject To
 Fraction_lower_bound_0: Fraction_for_second_order_amount[0] >= 0
 Fraction_upper_bound_0: Fraction_for_second_order_amount[0] <= 1
 Q0_upper_bound_0: - 2424.39646656199 Fraction_for_second_order_amount[0]
   + Q0_var[0] = 0
 Ensure_only_one_R_true_0: R[0,0] + R[0,1] + R[0,2] + R[0,3] + R[0,4]
   + R[0,5] + R[0,6] + R[0,7] = 1
 Define_Q_hat_0: - 2168.337106324953 R[0,0] - 2023.885856600318 R[0,1]
   - 2121.542305197294 R[0,2] - 2115.233664131875 R[0,3]
   - 2115.16068342389 R[0,4] - 2115.160683377049 R[0,5]
   - 2115.160684302211 R[0,6] - 2115.160681596941 R[0,7] + Q_hat[0] = 0
 Adjust_Q_hat_0: Q0_var[0] - Q_hat[0] + Q_hat_adjusted[0] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_0: Total_Demand_Up_to_K_minus_1[0]
   - 292.347948675354 R[0,0] - 502.3282049468224 R[0,1]
   - 771.7673366051861 R[0,2] - 981.7475919763692 R[0,3]
   - 1053.563279498847 R[0,4] - 1263.543534830728 R[0,5]
   - 1491.985623776599 R[0,6] - 1695.200170066968 R[0,7] = 0
 Constr_Total_Demand_from_K_to_T_0: Total_Demand_from_k_to_T[0]
   - 1822.812732055376 R[0,0] - 1612.832475783908 R[0,1]
   - 1343.393344125544 R[0,2] - 1133.413088754361 R[0,3]
   - 1061.597401231883 R[0,4] - 851.6171459000022 R[0,5]
   - 623.1750569541315 R[0,6] - 419.9605106637619 R[0,7] = 0
 Constr_Left_0_diff_aux_0: Sold_0[0] - Q0_var[0] + Left_0_aux_0 = 0
 Constr_Lost_0_diff_aux_0: - Total_Demand_Up_to_K_minus_1[0] + Q0_var[0]
   + Lost_0_aux_0 = 0
 Constr_Q1_plus_left_0: - Left_0[0] + Q1_plus_left[0] - Q1_var[0] = 0
 Constr_Left_1_diff_aux_0: Sold_1[0] - Q1_plus_left[0] + Left_0_aux_0 = 0
 Constr_Lost_1_diff_aux_0: - Total_Demand_from_k_to_T[0] + Q1_plus_left[0]
   + Lost_0_aux_0 = 0
 Profit_Constraint_0: - 700 Sold_0[0] + 700 Lost_0[0] - 700 Sold_1[0]
   + 300 Left_1[0] + 700 Lost_1[0] + profits_vars[0] = 0
 Fraction_lower_bound_1: Fraction_for_second_order_amount[1] >= 0
 Fraction_upper_bound_1: Fraction_for_second_order_amount[1] <= 1
 Q0_upper_bound_1: - 2424.39646656199 Fraction_for_second_order_amount[1]
   + Q0_var[1] = 0
 Ensure_only_one_R_true_1: R[1,0] + R[1,1] + R[1,2] + R[1,3] + R[1,4]
   + R[1,5] + R[1,6] + R[1,7] = 1
 Define_Q_hat_1: - 1969.699364242746 R[1,0] - 1005.214811168641 R[1,1]
   - 1029.032956510851 R[1,2] - 997.2906001503716 R[1,3]
   - 997.2175895781852 R[1,4] - 997.2175895313444 R[1,5]
   - 997.2175904565061 R[1,6] - 997.2175877512366 R[1,7] + Q_hat[1] = 0
 Adjust_Q_hat_1: Q0_var[1] - Q_hat[1] + Q_hat_adjusted[1] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_1: Total_Demand_Up_to_K_minus_1[1]
   - 142.9879872071073 R[1,0] - 228.8501994473938 R[1,1]
   - 375.7065207763131 R[1,2] - 461.5448432391966 R[1,3]
   - 461.5448432391966 R[1,4] - 547.3831650914601 R[1,5]
   - 649.4436979662599 R[1,6] - 825.5409431805019 R[1,7] = 0
 Constr_Total_Demand_from_K_to_T_1: Total_Demand_from_k_to_T[1]
   - 854.2295996779178 R[1,0] - 768.3673874376314 R[1,1]
   - 621.511066108712 R[1,2] - 535.6727436458286 R[1,3]
   - 535.6727436458286 R[1,4] - 449.8344217935651 R[1,5]
   - 347.7738889187651 R[1,6] - 171.6766437045232 R[1,7] = 0
 Constr_Left_0_diff_aux_1: Sold_0[1] - Q0_var[1] + Left_0_aux_1 = 0
 Constr_Lost_0_diff_aux_1: - Total_Demand_Up_to_K_minus_1[1] + Q0_var[1]
   + Lost_0_aux_1 = 0
 Constr_Q1_plus_left_1: - Left_0[1] + Q1_plus_left[1] - Q1_var[1] = 0
 Constr_Left_1_diff_aux_1: Sold_1[1] - Q1_plus_left[1] + Left_0_aux_1 = 0
 Constr_Lost_1_diff_aux_1: - Total_Demand_from_k_to_T[1] + Q1_plus_left[1]
   + Lost_0_aux_1 = 0
 Profit_Constraint_1: - 700 Sold_0[1] + 700 Lost_0[1] - 700 Sold_1[1]
   + 300 Left_1[1] + 700 Lost_1[1] + profits_vars[1] = 0
 Fraction_lower_bound_2: Fraction_for_second_order_amount[2] >= 0
 Fraction_upper_bound_2: Fraction_for_second_order_amount[2] <= 1
 Q0_upper_bound_2: - 2424.39646656199 Fraction_for_second_order_amount[2]
   + Q0_var[2] = 0
 Ensure_only_one_R_true_2: R[2,0] + R[2,1] + R[2,2] + R[2,3] + R[2,4]
   + R[2,5] + R[2,6] + R[2,7] = 1
 Define_Q_hat_2: - 1994.660788600626 R[2,0] - 2662.730137935359 R[2,1]
   - 2617.793137296004 R[2,2] - 2596.733261530278 R[2,3]
   - 2596.660305812767 R[2,4] - 2596.660305765926 R[2,5]
   - 2596.660306691087 R[2,6] - 2596.660303985818 R[2,7] + Q_hat[2] = 0
 Adjust_Q_hat_2: Q0_var[2] - Q_hat[2] + Q_hat_adjusted[2] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_2: Total_Demand_Up_to_K_minus_1[2]
   - 161.7570153948249 R[2,0] - 447.9242449579057 R[2,1]
   - 617.3596323728812 R[2,2] - 903.5268613063574 R[2,3]
   - 1142.883229127471 R[2,4] - 1429.050458389867 R[2,5]
   - 1728.060507212035 R[2,6] - 2024.325844594815 R[2,7] = 0
 Constr_Total_Demand_from_K_to_T_2: Total_Demand_from_k_to_T[2]
   - 2434.903287724782 R[2,0] - 2148.736058161701 R[2,1]
   - 1979.300670746726 R[2,2] - 1693.133441813249 R[2,3]
   - 1453.777073992136 R[2,4] - 1167.60984472974 R[2,5]
   - 868.5997959075721 R[2,6] - 572.3344585247919 R[2,7] = 0
 Constr_Left_0_diff_aux_2: Sold_0[2] - Q0_var[2] + Left_0_aux_2 = 0
 Constr_Lost_0_diff_aux_2: - Total_Demand_Up_to_K_minus_1[2] + Q0_var[2]
   + Lost_0_aux_2 = 0
 Constr_Q1_plus_left_2: - Left_0[2] + Q1_plus_left[2] - Q1_var[2] = 0
 Constr_Left_1_diff_aux_2: Sold_1[2] - Q1_plus_left[2] + Left_0_aux_2 = 0
 Constr_Lost_1_diff_aux_2: - Total_Demand_from_k_to_T[2] + Q1_plus_left[2]
   + Lost_0_aux_2 = 0
 Profit_Constraint_2: - 700 Sold_0[2] + 700 Lost_0[2] - 700 Sold_1[2]
   + 300 Left_1[2] + 700 Lost_1[2] + profits_vars[2] = 0
 Fraction_lower_bound_3: Fraction_for_second_order_amount[3] >= 0
 Fraction_upper_bound_3: Fraction_for_second_order_amount[3] <= 1
 Q0_upper_bound_3: - 2424.39646656199 Fraction_for_second_order_amount[3]
   + Q0_var[3] = 0
 Ensure_only_one_R_true_3: R[3,0] + R[3,1] + R[3,2] + R[3,3] + R[3,4]
   + R[3,5] + R[3,6] + R[3,7] = 1
 Define_Q_hat_3: - 2142.649031312608 R[3,0] - 1780.632853381889 R[3,1]
   - 1661.588033241784 R[3,2] - 1593.121539380112 R[3,3]
   - 1593.048560260281 R[3,4] - 1593.04856021344 R[3,5]
   - 1593.048561138601 R[3,6] - 1593.048558433332 R[3,7] + Q_hat[3] = 0
 Adjust_Q_hat_3: Q0_var[3] - Q_hat[3] + Q_hat_adjusted[3] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_3: Total_Demand_Up_to_K_minus_1[3]
   - 273.0325363382088 R[3,0] - 453.4945247993793 R[3,1]
   - 622.0231085204282 R[3,2] - 802.4851884717032 R[3,3]
   - 802.4851884717032 R[3,4] - 982.9472689092211 R[3,5]
   - 1119.771336574312 R[3,6] - 1232.124396692085 R[3,7] = 0
 Constr_Total_Demand_from_K_to_T_3: Total_Demand_from_k_to_T[3]
   - 1320.016021228912 R[3,0] - 1139.554032767742 R[3,1]
   - 971.0254490466926 R[3,2] - 790.5633690954177 R[3,3]
   - 790.5633690954177 R[3,4] - 610.1012886578998 R[3,5]
   - 473.2772209928088 R[3,6] - 360.9241608750359 R[3,7] = 0
 Constr_Left_0_diff_aux_3: Sold_0[3] - Q0_var[3] + Left_0_aux_3 = 0
 Constr_Lost_0_diff_aux_3: - Total_Demand_Up_to_K_minus_1[3] + Q0_var[3]
   + Lost_0_aux_3 = 0
 Constr_Q1_plus_left_3: - Left_0[3] + Q1_plus_left[3] - Q1_var[3] = 0
 Constr_Left_1_diff_aux_3: Sold_1[3] - Q1_plus_left[3] + Left_0_aux_3 = 0
 Constr_Lost_1_diff_aux_3: - Total_Demand_from_k_to_T[3] + Q1_plus_left[3]
   + Lost_0_aux_3 = 0
 Profit_Constraint_3: - 700 Sold_0[3] + 700 Lost_0[3] - 700 Sold_1[3]
   + 300 Left_1[3] + 700 Lost_1[3] + profits_vars[3] = 0
 Fraction_lower_bound_4: Fraction_for_second_order_amount[4] >= 0
 Fraction_upper_bound_4: Fraction_for_second_order_amount[4] <= 1
 Q0_upper_bound_4: - 2424.39646656199 Fraction_for_second_order_amount[4]
   + Q0_var[4] = 0
 Ensure_only_one_R_true_4: R[4,0] + R[4,1] + R[4,2] + R[4,3] + R[4,4]
   + R[4,5] + R[4,6] + R[4,7] = 1
 Define_Q_hat_4: - 2087.888092579251 R[4,0] - 1561.244769643312 R[4,1]
   - 1425.834565990179 R[4,2] - 1457.741643044815 R[4,3]
   - 1457.668634112587 R[4,4] - 1457.668634065746 R[4,5]
   - 1457.668634990908 R[4,6] - 1457.668632285638 R[4,7] + Q_hat[4] = 0
 Adjust_Q_hat_4: Q0_var[4] - Q_hat[4] + Q_hat_adjusted[4] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_4: Total_Demand_Up_to_K_minus_1[4]
   - 231.8566167297682 R[4,0] - 385.5213254516364 R[4,1]
   - 523.3741472040906 R[4,2] - 677.0396322047102 R[4,3]
   - 832.8123020350445 R[4,4] - 986.4777870326753 R[4,5]
   - 1140.057559323398 R[4,6] - 1150.337661424165 R[4,7] = 0
 Constr_Total_Demand_from_K_to_T_4: Total_Demand_from_k_to_T[4]
   - 1225.812014689659 R[4,0] - 1072.147305967791 R[4,1]
   - 934.2944842153363 R[4,2] - 780.6289992147168 R[4,3]
   - 624.8563293843824 R[4,4] - 471.1908443867516 R[4,5]
   - 317.6110720960293 R[4,6] - 307.330969995262 R[4,7] = 0
 Constr_Left_0_diff_aux_4: Sold_0[4] - Q0_var[4] + Left_0_aux_4 = 0
 Constr_Lost_0_diff_aux_4: - Total_Demand_Up_to_K_minus_1[4] + Q0_var[4]
   + Lost_0_aux_4 = 0
 Constr_Q1_plus_left_4: - Left_0[4] + Q1_plus_left[4] - Q1_var[4] = 0
 Constr_Left_1_diff_aux_4: Sold_1[4] - Q1_plus_left[4] + Left_0_aux_4 = 0
 Constr_Lost_1_diff_aux_4: - Total_Demand_from_k_to_T[4] + Q1_plus_left[4]
   + Lost_0_aux_4 = 0
 Profit_Constraint_4: - 700 Sold_0[4] + 700 Lost_0[4] - 700 Sold_1[4]
   + 300 Left_1[4] + 700 Lost_1[4] + profits_vars[4] = 0
Bounds
 profits_vars[0] free
 profits_vars[1] free
 profits_vars[2] free
 profits_vars[3] free
 profits_vars[4] free
 Fraction_for_second_order_amount[0] <= 1
 Fraction_for_second_order_amount[1] <= 1
 Fraction_for_second_order_amount[2] <= 1
 Fraction_for_second_order_amount[3] <= 1
 Fraction_for_second_order_amount[4] <= 1
 Q0_var[0] <= 2425.39646656199
 Q0_var[1] <= 2425.39646656199
 Q0_var[2] <= 2425.39646656199
 Q0_var[3] <= 2425.39646656199
 Q0_var[4] <= 2425.39646656199
 Q_hat_adjusted[0] free
 Q_hat_adjusted[1] free
 Q_hat_adjusted[2] free
 Q_hat_adjusted[3] free
 Q_hat_adjusted[4] free
 Left_0_aux_0 free
 Lost_0_aux_0 free
 Left_0_aux_0 free
 Lost_0_aux_0 free
 Left_0_aux_1 free
 Lost_0_aux_1 free
 Left_0_aux_1 free
 Lost_0_aux_1 free
 Left_0_aux_2 free
 Lost_0_aux_2 free
 Left_0_aux_2 free
 Lost_0_aux_2 free
 Left_0_aux_3 free
 Lost_0_aux_3 free
 Left_0_aux_3 free
 Lost_0_aux_3 free
 Left_0_aux_4 free
 Lost_0_aux_4 free
 Left_0_aux_4 free
 Lost_0_aux_4 free
Binaries
 R[0,0] R[0,1] R[0,2] R[0,3] R[0,4] R[0,5] R[0,6] R[0,7] R[1,0] R[1,1]
 R[1,2] R[1,3] R[1,4] R[1,5] R[1,6] R[1,7] R[2,0] R[2,1] R[2,2] R[2,3]
 R[2,4] R[2,5] R[2,6] R[2,7] R[3,0] R[3,1] R[3,2] R[3,3] R[3,4] R[3,5]
 R[3,6] R[3,7] R[4,0] R[4,1] R[4,2] R[4,3] R[4,4] R[4,5] R[4,6] R[4,7]
General Constraints
 Max_Constraint_0: Q1_var[0] = MAX ( Q_hat_adjusted[0] , 0 )
 Constr_Sold_0_min_0: Sold_0[0] = MIN ( Total_Demand_Up_to_K_minus_1[0] ,
   Q0_var[0] )
 Constr_Left_0_max_0: Left_0[0] = MAX ( Left_0_aux_0 , 0 )
 Constr_Lost_0_max_0: Lost_0[0] = MAX ( Lost_0_aux_0 , 0 )
 Constr_Sold_1_min_0: Sold_1[0] = MIN ( Total_Demand_from_k_to_T[0] ,
   Q1_plus_left[0] )
 Constr_Left_1_max_0: Left_1[0] = MAX ( Left_0_aux_0 , 0 )
 Constr_Lost_1_max_0: Lost_1[0] = MAX ( Lost_0_aux_0 , 0 )
 Max_Constraint_1: Q1_var[1] = MAX ( Q_hat_adjusted[1] , 0 )
 Constr_Sold_0_min_1: Sold_0[1] = MIN ( Total_Demand_Up_to_K_minus_1[1] ,
   Q0_var[1] )
 Constr_Left_0_max_1: Left_0[1] = MAX ( Left_0_aux_1 , 0 )
 Constr_Lost_0_max_1: Lost_0[1] = MAX ( Lost_0_aux_1 , 0 )
 Constr_Sold_1_min_1: Sold_1[1] = MIN ( Total_Demand_from_k_to_T[1] ,
   Q1_plus_left[1] )
 Constr_Left_1_max_1: Left_1[1] = MAX ( Left_0_aux_1 , 0 )
 Constr_Lost_1_max_1: Lost_1[1] = MAX ( Lost_0_aux_1 , 0 )
 Max_Constraint_2: Q1_var[2] = MAX ( Q_hat_adjusted[2] , 0 )
 Constr_Sold_0_min_2: Sold_0[2] = MIN ( Total_Demand_Up_to_K_minus_1[2] ,
   Q0_var[2] )
 Constr_Left_0_max_2: Left_0[2] = MAX ( Left_0_aux_2 , 0 )
 Constr_Lost_0_max_2: Lost_0[2] = MAX ( Lost_0_aux_2 , 0 )
 Constr_Sold_1_min_2: Sold_1[2] = MIN ( Total_Demand_from_k_to_T[2] ,
   Q1_plus_left[2] )
 Constr_Left_1_max_2: Left_1[2] = MAX ( Left_0_aux_2 , 0 )
 Constr_Lost_1_max_2: Lost_1[2] = MAX ( Lost_0_aux_2 , 0 )
 Max_Constraint_3: Q1_var[3] = MAX ( Q_hat_adjusted[3] , 0 )
 Constr_Sold_0_min_3: Sold_0[3] = MIN ( Total_Demand_Up_to_K_minus_1[3] ,
   Q0_var[3] )
 Constr_Left_0_max_3: Left_0[3] = MAX ( Left_0_aux_3 , 0 )
 Constr_Lost_0_max_3: Lost_0[3] = MAX ( Lost_0_aux_3 , 0 )
 Constr_Sold_1_min_3: Sold_1[3] = MIN ( Total_Demand_from_k_to_T[3] ,
   Q1_plus_left[3] )
 Constr_Left_1_max_3: Left_1[3] = MAX ( Left_0_aux_3 , 0 )
 Constr_Lost_1_max_3: Lost_1[3] = MAX ( Lost_0_aux_3 , 0 )
 Max_Constraint_4: Q1_var[4] = MAX ( Q_hat_adjusted[4] , 0 )
 Constr_Sold_0_min_4: Sold_0[4] = MIN ( Total_Demand_Up_to_K_minus_1[4] ,
   Q0_var[4] )
 Constr_Left_0_max_4: Left_0[4] = MAX ( Left_0_aux_4 , 0 )
 Constr_Lost_0_max_4: Lost_0[4] = MAX ( Lost_0_aux_4 , 0 )
 Constr_Sold_1_min_4: Sold_1[4] = MIN ( Total_Demand_from_k_to_T[4] ,
   Q1_plus_left[4] )
 Constr_Left_1_max_4: Left_1[4] = MAX ( Left_0_aux_4 , 0 )
 Constr_Lost_1_max_4: Lost_1[4] = MAX ( Lost_0_aux_4 , 0 )
End
