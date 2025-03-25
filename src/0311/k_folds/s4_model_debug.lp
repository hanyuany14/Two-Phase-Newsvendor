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
   + profits_vars[9]
Subject To
 Fraction_lower_bound_0: Fraction_for_second_order_amount[0] >= 0
 Fraction_upper_bound_0: Fraction_for_second_order_amount[0] <= 1
 Q0_upper_bound_0: - 2419.10426475367 Fraction_for_second_order_amount[0]
   + Q0_var[0] = 0
 Ensure_only_one_R_true_0: R[0,0] + R[0,1] + R[0,2] + R[0,3] + R[0,4]
   + R[0,5] + R[0,6] + R[0,7] = 1
 Define_Q_hat_0: - 1550.098651472731 R[0,0] - 1838.587436790646 R[0,1]
   - 1807.510558107486 R[0,2] - 1782.508727870664 R[0,3]
   - 1658.474600699251 R[0,4] - 1658.474674960262 R[0,5]
   - 1650.175403425489 R[0,6] - 1605.573885095132 R[0,7] + Q_hat[0] = 0
 Adjust_Q_hat_0: Q0_var[0] - Q_hat[0] + Q_hat_adjusted[0] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_0: Total_Demand_Up_to_K_minus_1[0]
   - 78.22355461859897 R[0,0] - 258.4827174146135 R[0,1]
   - 394.1951397570832 R[0,2] - 574.4542910844115 R[0,3]
   - 750.9173064868285 R[0,4] - 931.1764580378608 R[0,5]
   - 1089.650398684325 R[0,6] - 1245.055579433237 R[0,7] = 0
 Constr_Total_Demand_from_K_to_T_0: Total_Demand_from_k_to_T[0]
   - 1527.350327916703 R[0,0] - 1347.091165120688 R[0,1]
   - 1211.378742778218 R[0,2] - 1031.11959145089 R[0,3]
   - 854.6565760484732 R[0,4] - 674.3974244974409 R[0,5]
   - 515.923483850977 R[0,6] - 360.5183031020646 R[0,7] = 0
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
 Q0_upper_bound_1: - 2419.10426475367 Fraction_for_second_order_amount[1]
   + Q0_var[1] = 0
 Ensure_only_one_R_true_1: R[1,0] + R[1,1] + R[1,2] + R[1,3] + R[1,4]
   + R[1,5] + R[1,6] + R[1,7] = 1
 Define_Q_hat_1: - 1462.707395390741 R[1,0] - 751.395511538539 R[1,1]
   - 828.2885738201738 R[1,2] - 804.256494864439 R[1,3]
   - 776.8952620734292 R[1,4] - 776.8960543086139 R[1,5]
   - 797.450665649649 R[1,6] - 787.5786973800305 R[1,7] + Q_hat[1] = 0
 Adjust_Q_hat_1: Q0_var[1] - Q_hat[1] + Q_hat_adjusted[1] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_1: Total_Demand_Up_to_K_minus_1[1]
   - 61.09264422377667 R[1,0] - 124.2653534747971 R[1,1]
   - 124.2653534747971 R[1,2] - 187.8498507163212 R[1,3]
   - 192.3393536915355 R[1,4] - 255.9238507763015 R[1,5]
   - 571.6544719164667 R[1,6] - 660.4097048264515 R[1,7] = 0
 Constr_Total_Demand_from_K_to_T_1: Total_Demand_from_k_to_T[1]
   - 726.4860548932908 R[1,0] - 663.3133456422704 R[1,1]
   - 663.3133456422704 R[1,2] - 599.7288484007463 R[1,3]
   - 595.239345425532 R[1,4] - 531.654848340766 R[1,5]
   - 215.9242272006008 R[1,6] - 127.168994290616 R[1,7] = 0
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
 Q0_upper_bound_2: - 2419.10426475367 Fraction_for_second_order_amount[2]
   + Q0_var[2] = 0
 Ensure_only_one_R_true_2: R[2,0] + R[2,1] + R[2,2] + R[2,3] + R[2,4]
   + R[2,5] + R[2,6] + R[2,7] = 1
 Define_Q_hat_2: - 2036.567028442724 R[2,0] - 1227.23704286003 R[2,1]
   - 1255.644487984717 R[2,2] - 1313.627020570647 R[2,3]
   - 1322.722959198845 R[2,4] - 1322.721815366394 R[2,5]
   - 1303.182888383296 R[2,6] - 1316.217687043345 R[2,7] + Q_hat[2] = 0
 Adjust_Q_hat_2: Q0_var[2] - Q_hat[2] + Q_hat_adjusted[2] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_2: Total_Demand_Up_to_K_minus_1[2]
   - 173.5837376553206 R[2,0] - 273.6854496451681 R[2,1]
   - 442.2244706453832 R[2,2] - 542.2236021034597 R[2,3]
   - 617.2315810442274 R[2,4] - 717.2307122683279 R[2,5]
   - 855.9827410974402 R[2,6] - 1116.219420694144 R[2,7] = 0
 Constr_Total_Demand_from_K_to_T_2: Total_Demand_from_k_to_T[2]
   - 1142.633945486942 R[2,0] - 1042.532233497095 R[2,1]
   - 873.9932124968796 R[2,2] - 773.9940810388032 R[2,3]
   - 698.9861020980354 R[2,4] - 598.986970873935 R[2,5]
   - 460.2349420448227 R[2,6] - 199.9982624481188 R[2,7] = 0
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
 Q0_upper_bound_3: - 2419.10426475367 Fraction_for_second_order_amount[3]
   + Q0_var[3] = 0
 Ensure_only_one_R_true_3: R[3,0] + R[3,1] + R[3,2] + R[3,3] + R[3,4]
   + R[3,5] + R[3,6] + R[3,7] = 1
 Define_Q_hat_3: - 1823.847974216397 R[3,0] - 769.9265792761769 R[3,1]
   - 635.6563549055404 R[3,2] - 683.0523571314748 R[3,3]
   - 673.9806688337958 R[3,4] - 673.9812624338404 R[3,5]
   - 676.6853511144207 R[3,6] - 651.4438298532222 R[3,7] + Q_hat[3] = 0
 Adjust_Q_hat_3: Q0_var[3] - Q_hat[3] + Q_hat_adjusted[3] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_3: Total_Demand_Up_to_K_minus_1[3]
   - 131.8853890950972 R[3,0] - 187.440040187972 R[3,1]
   - 432.9220706793407 R[3,2] - 487.5525193594574 R[3,3]
   - 487.5525193594574 R[3,4] - 542.1829342806082 R[3,5]
   - 542.1829342806082 R[3,6] - 542.1829342806082 R[3,7] = 0
 Constr_Total_Demand_from_K_to_T_3: Total_Demand_from_k_to_T[3]
   - 519.5584424159283 R[3,0] - 464.0037913230535 R[3,1]
   - 218.5217608316849 R[3,2] - 163.8913121515681 R[3,3]
   - 163.8913121515681 R[3,4] - 109.2608972304174 R[3,5]
   - 109.2608972304174 R[3,6] - 109.2608972304174 R[3,7] = 0
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
 Q0_upper_bound_4: - 2419.10426475367 Fraction_for_second_order_amount[4]
   + Q0_var[4] = 0
 Ensure_only_one_R_true_4: R[4,0] + R[4,1] + R[4,2] + R[4,3] + R[4,4]
   + R[4,5] + R[4,6] + R[4,7] = 1
 Define_Q_hat_4: - 2323.165135541887 R[4,0] - 2647.273095942448 R[4,1]
   - 2622.505931004775 R[4,2] - 2580.861435867828 R[4,3]
   - 2685.196820407801 R[4,4] - 2685.196912125295 R[4,5]
   - 2694.532068269726 R[4,6] - 2665.493954136803 R[4,7] + Q_hat[4] = 0
 Adjust_Q_hat_4: Q0_var[4] - Q_hat[4] + Q_hat_adjusted[4] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_4: Total_Demand_Up_to_K_minus_1[4]
   - 229.7642608810676 R[4,0] - 478.1886869613019 R[4,1]
   - 787.2259869290319 R[4,2] - 1035.650412700505 R[4,3]
   - 1423.775438975507 R[4,4] - 1672.199864814862 R[4,5]
   - 1856.406719775763 R[4,6] - 2168.645098800159 R[4,7] = 0
 Constr_Total_Demand_from_K_to_T_4: Total_Demand_from_k_to_T[4]
   - 2435.729689597802 R[4,0] - 2187.305263517567 R[4,1]
   - 1878.267963549837 R[4,2] - 1629.843537778364 R[4,3]
   - 1241.718511503363 R[4,4] - 993.2940856640074 R[4,5]
   - 809.0872307031066 R[4,6] - 496.8488516787103 R[4,7] = 0
 Constr_Left_0_diff_aux_4: Sold_0[4] - Q0_var[4] + Left_0_aux_4 = 0
 Constr_Lost_0_diff_aux_4: - Total_Demand_Up_to_K_minus_1[4] + Q0_var[4]
   + Lost_0_aux_4 = 0
 Constr_Q1_plus_left_4: - Left_0[4] + Q1_plus_left[4] - Q1_var[4] = 0
 Constr_Left_1_diff_aux_4: Sold_1[4] - Q1_plus_left[4] + Left_0_aux_4 = 0
 Constr_Lost_1_diff_aux_4: - Total_Demand_from_k_to_T[4] + Q1_plus_left[4]
   + Lost_0_aux_4 = 0
 Profit_Constraint_4: - 700 Sold_0[4] + 700 Lost_0[4] - 700 Sold_1[4]
   + 300 Left_1[4] + 700 Lost_1[4] + profits_vars[4] = 0
 Fraction_lower_bound_5: Fraction_for_second_order_amount[5] >= 0
 Fraction_upper_bound_5: Fraction_for_second_order_amount[5] <= 1
 Q0_upper_bound_5: - 2419.10426475367 Fraction_for_second_order_amount[5]
   + Q0_var[5] = 0
 Ensure_only_one_R_true_5: R[5,0] + R[5,1] + R[5,2] + R[5,3] + R[5,4]
   + R[5,5] + R[5,6] + R[5,7] = 1
 Define_Q_hat_5: - 1270.457972116784 R[5,0] - 1094.242445556085 R[5,1]
   - 1025.890319523614 R[5,2] - 899.6273904401014 R[5,3]
   - 856.5555636506734 R[5,4] - 856.5550626425813 R[5,5]
   - 857.1824002942093 R[5,6] - 887.027648283059 R[5,7] + Q_hat[5] = 0
 Adjust_Q_hat_5: Q0_var[5] - Q_hat[5] + Q_hat_adjusted[5] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_5: Total_Demand_Up_to_K_minus_1[5]
   - 23.40686430464427 R[5,0] - 129.3785579282645 R[5,1]
   - 218.3995788103484 R[5,2] - 324.3807511415336 R[5,3]
   - 409.417399450705 R[5,4] - 515.3985714658 R[5,5]
   - 608.3824253224716 R[5,6] - 675.0653023803645 R[5,7] = 0
 Constr_Total_Demand_from_K_to_T_5: Total_Demand_from_k_to_T[5]
   - 863.6207821059104 R[5,0] - 757.6490884822902 R[5,1]
   - 668.6280676002064 R[5,2] - 562.6468952690211 R[5,3]
   - 477.6102469598496 R[5,4] - 371.6290749447546 R[5,5]
   - 278.6452210880831 R[5,6] - 211.9623440301901 R[5,7] = 0
 Constr_Left_0_diff_aux_5: Sold_0[5] - Q0_var[5] + Left_0_aux_5 = 0
 Constr_Lost_0_diff_aux_5: - Total_Demand_Up_to_K_minus_1[5] + Q0_var[5]
   + Lost_0_aux_5 = 0
 Constr_Q1_plus_left_5: - Left_0[5] + Q1_plus_left[5] - Q1_var[5] = 0
 Constr_Left_1_diff_aux_5: Sold_1[5] - Q1_plus_left[5] + Left_0_aux_5 = 0
 Constr_Lost_1_diff_aux_5: - Total_Demand_from_k_to_T[5] + Q1_plus_left[5]
   + Lost_0_aux_5 = 0
 Profit_Constraint_5: - 700 Sold_0[5] + 700 Lost_0[5] - 700 Sold_1[5]
   + 300 Left_1[5] + 700 Lost_1[5] + profits_vars[5] = 0
 Fraction_lower_bound_6: Fraction_for_second_order_amount[6] >= 0
 Fraction_upper_bound_6: Fraction_for_second_order_amount[6] <= 1
 Q0_upper_bound_6: - 2419.10426475367 Fraction_for_second_order_amount[6]
   + Q0_var[6] = 0
 Ensure_only_one_R_true_6: R[6,0] + R[6,1] + R[6,2] + R[6,3] + R[6,4]
   + R[6,5] + R[6,6] + R[6,7] = 1
 Define_Q_hat_6: - 1151.050700385223 R[6,0] - 1341.58286460118 R[6,1]
   - 1361.301082349022 R[6,2] - 1454.128103770024 R[6,3]
   - 1406.177994654571 R[6,4] - 1406.17778960854 R[6,5]
   - 1411.436310986697 R[6,6] - 1401.460988594132 R[6,7] + Q_hat[6] = 0
 Adjust_Q_hat_6: Q0_var[6] - Q_hat[6] + Q_hat_adjusted[6] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_6: Total_Demand_Up_to_K_minus_1[6]
   - 136.3326395466707 R[6,1] - 136.3326395466707 R[6,2]
   - 272.6705595886043 R[6,3] - 445.646590542891 R[6,4]
   - 581.9845107171167 R[6,5] - 883.7445453613815 R[6,6]
   - 1128.785146254627 R[6,7] = 0
 Constr_Total_Demand_from_K_to_T_6: Total_Demand_from_k_to_T[6]
   - 1401.460986603079 R[6,0] - 1265.128347056409 R[6,1]
   - 1265.128347056409 R[6,2] - 1128.790427014475 R[6,3]
   - 955.8143960601883 R[6,4] - 819.4764758859626 R[6,5]
   - 517.7164412416977 R[6,6] - 272.6758403484523 R[6,7] = 0
 Constr_Left_0_diff_aux_6: Sold_0[6] - Q0_var[6] + Left_0_aux_6 = 0
 Constr_Lost_0_diff_aux_6: - Total_Demand_Up_to_K_minus_1[6] + Q0_var[6]
   + Lost_0_aux_6 = 0
 Constr_Q1_plus_left_6: - Left_0[6] + Q1_plus_left[6] - Q1_var[6] = 0
 Constr_Left_1_diff_aux_6: Sold_1[6] - Q1_plus_left[6] + Left_0_aux_6 = 0
 Constr_Lost_1_diff_aux_6: - Total_Demand_from_k_to_T[6] + Q1_plus_left[6]
   + Lost_0_aux_6 = 0
 Profit_Constraint_6: - 700 Sold_0[6] + 700 Lost_0[6] - 700 Sold_1[6]
   + 300 Left_1[6] + 700 Lost_1[6] + profits_vars[6] = 0
 Fraction_lower_bound_7: Fraction_for_second_order_amount[7] >= 0
 Fraction_upper_bound_7: Fraction_for_second_order_amount[7] <= 1
 Q0_upper_bound_7: - 2419.10426475367 Fraction_for_second_order_amount[7]
   + Q0_var[7] = 0
 Ensure_only_one_R_true_7: R[7,0] + R[7,1] + R[7,2] + R[7,3] + R[7,4]
   + R[7,5] + R[7,6] + R[7,7] = 1
 Define_Q_hat_7: - 2289.367962295753 R[7,0] - 2944.896992688336 R[7,1]
   - 2935.543832289447 R[7,2] - 2948.998827082371 R[7,3]
   - 2940.830213820026 R[7,4] - 2940.830757023105 R[7,5]
   - 2930.350361128611 R[7,6] - 2936.077218398539 R[7,7] + Q_hat[7] = 0
 Adjust_Q_hat_7: Q0_var[7] - Q_hat[7] + Q_hat_adjusted[7] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_7: Total_Demand_Up_to_K_minus_1[7]
   - 223.1391547499163 R[7,0] - 505.1594784200425 R[7,1]
   - 803.0524238373333 R[7,2] - 1085.072746985389 R[7,3]
   - 1482.459653754254 R[7,4] - 1764.479977120651 R[7,5]
   - 1947.039041521234 R[7,6] - 2372.036576727909 R[7,7] = 0
 Constr_Total_Demand_from_K_to_T_7: Total_Demand_from_k_to_T[7]
   - 2712.938068710788 R[7,0] - 2430.917745040662 R[7,1]
   - 2133.024799623371 R[7,2] - 1851.004476475316 R[7,3]
   - 1453.61756970645 R[7,4] - 1171.597246340053 R[7,5]
   - 989.0381819394702 R[7,6] - 564.0406467327954 R[7,7] = 0
 Constr_Left_0_diff_aux_7: Sold_0[7] - Q0_var[7] + Left_0_aux_7 = 0
 Constr_Lost_0_diff_aux_7: - Total_Demand_Up_to_K_minus_1[7] + Q0_var[7]
   + Lost_0_aux_7 = 0
 Constr_Q1_plus_left_7: - Left_0[7] + Q1_plus_left[7] - Q1_var[7] = 0
 Constr_Left_1_diff_aux_7: Sold_1[7] - Q1_plus_left[7] + Left_0_aux_7 = 0
 Constr_Lost_1_diff_aux_7: - Total_Demand_from_k_to_T[7] + Q1_plus_left[7]
   + Lost_0_aux_7 = 0
 Profit_Constraint_7: - 700 Sold_0[7] + 700 Lost_0[7] - 700 Sold_1[7]
   + 300 Left_1[7] + 700 Lost_1[7] + profits_vars[7] = 0
 Fraction_lower_bound_8: Fraction_for_second_order_amount[8] >= 0
 Fraction_upper_bound_8: Fraction_for_second_order_amount[8] <= 1
 Q0_upper_bound_8: - 2419.10426475367 Fraction_for_second_order_amount[8]
   + Q0_var[8] = 0
 Ensure_only_one_R_true_8: R[8,0] + R[8,1] + R[8,2] + R[8,3] + R[8,4]
   + R[8,5] + R[8,6] + R[8,7] = 1
 Define_Q_hat_8: - 3253.175019539186 R[8,0] - 2670.388927889571 R[8,1]
   - 2686.129544782991 R[8,2] - 2636.630532940255 R[8,3]
   - 2535.86118041466 R[8,4] - 2535.861277955109 R[8,5]
   - 2538.131781908193 R[8,6] - 2524.272355234561 R[8,7] + Q_hat[8] = 0
 Adjust_Q_hat_8: Q0_var[8] - Q_hat[8] + Q_hat_adjusted[8] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_8: Total_Demand_Up_to_K_minus_1[8]
   - 412.0698679100998 R[8,0] - 638.1734778075961 R[8,1]
   - 1102.612355542231 R[8,2] - 1328.715956101339 R[8,3]
   - 1494.701001342785 R[8,4] - 1720.804601823668 R[8,5]
   - 1844.546426382748 R[8,6] - 2072.065153132653 R[8,7] = 0
 Constr_Total_Demand_from_K_to_T_8: Total_Demand_from_k_to_T[8]
   - 2112.20248618432 R[8,0] - 1886.098876286824 R[8,1]
   - 1421.659998552189 R[8,2] - 1195.556397993081 R[8,3]
   - 1029.571352751635 R[8,4] - 803.4677522707516 R[8,5]
   - 679.7259277116714 R[8,6] - 452.2072009617664 R[8,7] = 0
 Constr_Left_0_diff_aux_8: Sold_0[8] - Q0_var[8] + Left_0_aux_8 = 0
 Constr_Lost_0_diff_aux_8: - Total_Demand_Up_to_K_minus_1[8] + Q0_var[8]
   + Lost_0_aux_8 = 0
 Constr_Q1_plus_left_8: - Left_0[8] + Q1_plus_left[8] - Q1_var[8] = 0
 Constr_Left_1_diff_aux_8: Sold_1[8] - Q1_plus_left[8] + Left_0_aux_8 = 0
 Constr_Lost_1_diff_aux_8: - Total_Demand_from_k_to_T[8] + Q1_plus_left[8]
   + Lost_0_aux_8 = 0
 Profit_Constraint_8: - 700 Sold_0[8] + 700 Lost_0[8] - 700 Sold_1[8]
   + 300 Left_1[8] + 700 Lost_1[8] + profits_vars[8] = 0
 Fraction_lower_bound_9: Fraction_for_second_order_amount[9] >= 0
 Fraction_upper_bound_9: Fraction_for_second_order_amount[9] <= 1
 Q0_upper_bound_9: - 2419.10426475367 Fraction_for_second_order_amount[9]
   + Q0_var[9] = 0
 Ensure_only_one_R_true_9: R[9,0] + R[9,1] + R[9,2] + R[9,3] + R[9,4]
   + R[9,5] + R[9,6] + R[9,7] = 1
 Define_Q_hat_9: - 1317.741154578166 R[9,0] - 665.0440559039114 R[9,1]
   - 703.0824232994238 R[9,2] - 611.225172204287 R[9,3]
   - 633.9549210164575 R[9,4] - 633.9554781070678 R[9,5]
   - 614.6692315100746 R[9,6] - 575.2654344444128 R[9,7] + Q_hat[9] = 0
 Adjust_Q_hat_9: Q0_var[9] - Q_hat[9] + Q_hat_adjusted[9] = 0
 Constr_Total_Demand_Up_to_K_Minus_1_9: Total_Demand_Up_to_K_minus_1[9]
   - 32.67557147562292 R[9,0] - 90.2380865972281 R[9,1]
   - 90.2380865972281 R[9,2] - 148.1978188861405 R[9,3]
   - 172.6532503503669 R[9,4] - 230.6129827360392 R[9,5]
   - 373.0684307997718 R[9,6] - 459.345971061528 R[9,7] = 0
 Constr_Total_Demand_from_K_to_T_9: Total_Demand_from_k_to_T[9]
   - 542.5898643768705 R[9,0] - 485.0273492552654 R[9,1]
   - 485.0273492552654 R[9,2] - 427.067616966353 R[9,3]
   - 402.6121855021266 R[9,4] - 344.6524531164544 R[9,5]
   - 202.1970050527217 R[9,6] - 115.9194647909655 R[9,7] = 0
 Constr_Left_0_diff_aux_9: Sold_0[9] - Q0_var[9] + Left_0_aux_9 = 0
 Constr_Lost_0_diff_aux_9: - Total_Demand_Up_to_K_minus_1[9] + Q0_var[9]
   + Lost_0_aux_9 = 0
 Constr_Q1_plus_left_9: - Left_0[9] + Q1_plus_left[9] - Q1_var[9] = 0
 Constr_Left_1_diff_aux_9: Sold_1[9] - Q1_plus_left[9] + Left_0_aux_9 = 0
 Constr_Lost_1_diff_aux_9: - Total_Demand_from_k_to_T[9] + Q1_plus_left[9]
   + Lost_0_aux_9 = 0
 Profit_Constraint_9: - 700 Sold_0[9] + 700 Lost_0[9] - 700 Sold_1[9]
   + 300 Left_1[9] + 700 Lost_1[9] + profits_vars[9] = 0
Bounds
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
 Q0_var[0] <= 2420.10426475367
 Q0_var[1] <= 2420.10426475367
 Q0_var[2] <= 2420.10426475367
 Q0_var[3] <= 2420.10426475367
 Q0_var[4] <= 2420.10426475367
 Q0_var[5] <= 2420.10426475367
 Q0_var[6] <= 2420.10426475367
 Q0_var[7] <= 2420.10426475367
 Q0_var[8] <= 2420.10426475367
 Q0_var[9] <= 2420.10426475367
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
 Left_0_aux_5 free
 Lost_0_aux_5 free
 Left_0_aux_5 free
 Lost_0_aux_5 free
 Left_0_aux_6 free
 Lost_0_aux_6 free
 Left_0_aux_6 free
 Lost_0_aux_6 free
 Left_0_aux_7 free
 Lost_0_aux_7 free
 Left_0_aux_7 free
 Lost_0_aux_7 free
 Left_0_aux_8 free
 Lost_0_aux_8 free
 Left_0_aux_8 free
 Lost_0_aux_8 free
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
 Max_Constraint_5: Q1_var[5] = MAX ( Q_hat_adjusted[5] , 0 )
 Constr_Sold_0_min_5: Sold_0[5] = MIN ( Total_Demand_Up_to_K_minus_1[5] ,
   Q0_var[5] )
 Constr_Left_0_max_5: Left_0[5] = MAX ( Left_0_aux_5 , 0 )
 Constr_Lost_0_max_5: Lost_0[5] = MAX ( Lost_0_aux_5 , 0 )
 Constr_Sold_1_min_5: Sold_1[5] = MIN ( Total_Demand_from_k_to_T[5] ,
   Q1_plus_left[5] )
 Constr_Left_1_max_5: Left_1[5] = MAX ( Left_0_aux_5 , 0 )
 Constr_Lost_1_max_5: Lost_1[5] = MAX ( Lost_0_aux_5 , 0 )
 Max_Constraint_6: Q1_var[6] = MAX ( Q_hat_adjusted[6] , 0 )
 Constr_Sold_0_min_6: Sold_0[6] = MIN ( Total_Demand_Up_to_K_minus_1[6] ,
   Q0_var[6] )
 Constr_Left_0_max_6: Left_0[6] = MAX ( Left_0_aux_6 , 0 )
 Constr_Lost_0_max_6: Lost_0[6] = MAX ( Lost_0_aux_6 , 0 )
 Constr_Sold_1_min_6: Sold_1[6] = MIN ( Total_Demand_from_k_to_T[6] ,
   Q1_plus_left[6] )
 Constr_Left_1_max_6: Left_1[6] = MAX ( Left_0_aux_6 , 0 )
 Constr_Lost_1_max_6: Lost_1[6] = MAX ( Lost_0_aux_6 , 0 )
 Max_Constraint_7: Q1_var[7] = MAX ( Q_hat_adjusted[7] , 0 )
 Constr_Sold_0_min_7: Sold_0[7] = MIN ( Total_Demand_Up_to_K_minus_1[7] ,
   Q0_var[7] )
 Constr_Left_0_max_7: Left_0[7] = MAX ( Left_0_aux_7 , 0 )
 Constr_Lost_0_max_7: Lost_0[7] = MAX ( Lost_0_aux_7 , 0 )
 Constr_Sold_1_min_7: Sold_1[7] = MIN ( Total_Demand_from_k_to_T[7] ,
   Q1_plus_left[7] )
 Constr_Left_1_max_7: Left_1[7] = MAX ( Left_0_aux_7 , 0 )
 Constr_Lost_1_max_7: Lost_1[7] = MAX ( Lost_0_aux_7 , 0 )
 Max_Constraint_8: Q1_var[8] = MAX ( Q_hat_adjusted[8] , 0 )
 Constr_Sold_0_min_8: Sold_0[8] = MIN ( Total_Demand_Up_to_K_minus_1[8] ,
   Q0_var[8] )
 Constr_Left_0_max_8: Left_0[8] = MAX ( Left_0_aux_8 , 0 )
 Constr_Lost_0_max_8: Lost_0[8] = MAX ( Lost_0_aux_8 , 0 )
 Constr_Sold_1_min_8: Sold_1[8] = MIN ( Total_Demand_from_k_to_T[8] ,
   Q1_plus_left[8] )
 Constr_Left_1_max_8: Left_1[8] = MAX ( Left_0_aux_8 , 0 )
 Constr_Lost_1_max_8: Lost_1[8] = MAX ( Lost_0_aux_8 , 0 )
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
