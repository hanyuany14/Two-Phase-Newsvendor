\ Model profit_maximization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  profits_vars[0] + profits_vars[1] + profits_vars[2] + profits_vars[3]
   + profits_vars[4] + profits_vars[5] + profits_vars[6] + profits_vars[7]
   + profits_vars[8] + profits_vars[9]
Subject To
 Fraction_lower_bound_0: Fraction[0] >= 0
 Fraction_upper_bound_0: Fraction[0] <= 1
 R2: 0.1456017588547455 alphas[0] - 0.3790439810995431 alphas[1]
   - 0.3716393994299632 alphas[2] + 0.00107812104660135 alphas[3]
   - alphas[4] + f_var[0] = 0
 R3: - 2419.10426475367 Fraction[0] + Q0_var[0] = 0
 R4: Q0_var[0] + Lost_0_expr_0 = 1758.349487415308
 R5: Q0_var[0] + Q_hat_adjusted_0 = 2207.064385470138
 R6: Sold_0[0] - Q0_var[0] - Q1_var[0] + total_stock_second_period_0 = 0
 R7: total_stock_second_period_0 + Lost_1_expr_0 = 448.7143865722818
 R8: Sold_1[0] + Left_1[0] - total_stock_second_period_0 = 0
 R9: - 700 Sold_0[0] - 700 Sold_1[0] + 700 Lost_0[0] + 700 Lost_1[0]
   + 300 Left_1[0] + profits_vars[0] = 0
 Fraction_lower_bound_1: Fraction[1] >= 0
 Fraction_upper_bound_1: Fraction[1] <= 1
 R12: - 0.2125425940893114 alphas[0] + 1.21795628154855 alphas[1]
   + 0.4302928522738437 alphas[2] - 0.8594343767013294 alphas[3]
   - alphas[4] + f_var[1] = 0
 R13: - 2419.10426475367 Fraction[1] + Q0_var[1] = 0
 R14: Q0_var[1] + Lost_0_expr_1 = 1255.034333956949
 R15: Q0_var[1] + Q_hat_adjusted_1 = 1581.806186678592
 R16: Sold_0[1] - Q0_var[1] - Q1_var[1] + total_stock_second_period_1 = 0
 R17: total_stock_second_period_1 + Lost_1_expr_1 = 326.7713413390344
 R18: Sold_1[1] + Left_1[1] - total_stock_second_period_1 = 0
 R19: - 700 Sold_0[1] - 700 Sold_1[1] + 700 Lost_0[1] + 700 Lost_1[1]
   + 300 Left_1[1] + profits_vars[1] = 0
 Fraction_lower_bound_2: Fraction[2] >= 0
 Fraction_upper_bound_2: Fraction[2] <= 1
 R22: - 1.416454709959566 alphas[0] - 0.6037372987839048 alphas[1]
   - 0.4526159169918287 alphas[2] + 1.256308100289521 alphas[3] - alphas[4]
   + f_var[2] = 0
 R23: - 2419.10426475367 Fraction[2] + Q0_var[2] = 0
 R24: Q0_var[2] + Lost_0_expr_2 = 2218.733264564339
 R25: Q0_var[2] + Q_hat_adjusted_2 = 2679.761575663252
 R26: Sold_0[2] - Q0_var[2] - Q1_var[2] + total_stock_second_period_2 = 0
 R27: total_stock_second_period_2 + Lost_1_expr_2 = 461.027799735174
 R28: Sold_1[2] + Left_1[2] - total_stock_second_period_2 = 0
 R29: - 700 Sold_0[2] - 700 Sold_1[2] + 700 Lost_0[2] + 700 Lost_1[2]
   + 300 Left_1[2] + profits_vars[2] = 0
 Fraction_lower_bound_3: Fraction[3] >= 0
 Fraction_upper_bound_3: Fraction[3] <= 1
 R32: 1.528221828396282 alphas[0] - 1.49266005616474 alphas[1]
   - 0.9271827793673989 alphas[2] - 0.5959721784765007 alphas[3]
   - alphas[4] + f_var[3] = 0
 R33: - 2419.10426475367 Fraction[3] + Q0_var[3] = 0
 R34: Q0_var[3] + Lost_0_expr_3 = 2075.062390905664
 R35: Q0_var[3] + Q_hat_adjusted_3 = 2608.254065136769
 R36: Sold_0[3] - Q0_var[3] - Q1_var[3] + total_stock_second_period_3 = 0
 R37: total_stock_second_period_3 + Lost_1_expr_3 = 533.1911629643146
 R38: Sold_1[3] + Left_1[3] - total_stock_second_period_3 = 0
 R39: - 700 Sold_0[3] - 700 Sold_1[3] + 700 Lost_0[3] + 700 Lost_1[3]
   + 300 Left_1[3] + profits_vars[3] = 0
 Fraction_lower_bound_4: Fraction[4] >= 0
 Fraction_upper_bound_4: Fraction[4] <= 1
 R42: - 1.273867487915399 alphas[0] + 1.301191337401891 alphas[1]
   - 1.286047967032324 alphas[2] + 1.566747047873712 alphas[3] - alphas[4]
   + f_var[4] = 0
 R43: - 2419.10426475367 Fraction[4] + Q0_var[4] = 0
 R44: Q0_var[4] + Lost_0_expr_4 = 2413.242143776983
 R45: Q0_var[4] + Q_hat_adjusted_4 = 3001.003407723577
 R46: Sold_0[4] - Q0_var[4] - Q1_var[4] + total_stock_second_period_4 = 0
 R47: total_stock_second_period_4 + Lost_1_expr_4 = 587.7607525014429
 R48: Sold_1[4] + Left_1[4] - total_stock_second_period_4 = 0
 R49: - 700 Sold_0[4] - 700 Sold_1[4] + 700 Lost_0[4] + 700 Lost_1[4]
   + 300 Left_1[4] + profits_vars[4] = 0
 Fraction_lower_bound_5: Fraction[5] >= 0
 Fraction_upper_bound_5: Fraction[5] <= 1
 R52: - 1.291751382636692 alphas[0] + 0.0190589215567936 alphas[1]
   - 0.8923975725868508 alphas[2] - 1.142451037178154 alphas[3] - alphas[4]
   + f_var[5] = 0
 R53: - 2419.10426475367 Fraction[5] + Q0_var[5] = 0
 R54: Q0_var[5] + Lost_0_expr_5 = 1971.625493125927
 R55: Q0_var[5] + Q_hat_adjusted_5 = 2499.527675707747
 R56: Sold_0[5] - Q0_var[5] - Q1_var[5] + total_stock_second_period_5 = 0
 R57: total_stock_second_period_5 + Lost_1_expr_5 = 527.9016711963055
 R58: Sold_1[5] + Left_1[5] - total_stock_second_period_5 = 0
 R59: - 700 Sold_0[5] - 700 Sold_1[5] + 700 Lost_0[5] + 700 Lost_1[5]
   + 300 Left_1[5] + profits_vars[5] = 0
 Fraction_lower_bound_6: Fraction[6] >= 0
 Fraction_upper_bound_6: Fraction[6] <= 1
 R62: 0.8345285628671644 alphas[0] - 0.043146013826882 alphas[1]
   + 1.883088303539478 alphas[2] - 0.2055364072414201 alphas[3] - alphas[4]
   + f_var[6] = 0
 R63: - 2419.10426475367 Fraction[6] + Q0_var[6] = 0
 R64: Q0_var[6] + Lost_0_expr_6 = 569.4182127548495
 R65: Q0_var[6] + Q_hat_adjusted_6 = 675.2757661884438
 R66: Sold_0[6] - Q0_var[6] - Q1_var[6] + total_stock_second_period_6 = 0
 R67: total_stock_second_period_6 + Lost_1_expr_6 = 105.8570420498363
 R68: Sold_1[6] + Left_1[6] - total_stock_second_period_6 = 0
 R69: - 700 Sold_0[6] - 700 Sold_1[6] + 700 Lost_0[6] + 700 Lost_1[6]
   + 300 Left_1[6] + profits_vars[6] = 0
 Fraction_lower_bound_7: Fraction[7] >= 0
 Fraction_upper_bound_7: Fraction[7] <= 1
 R72: 1.210838876139293 alphas[0] + 0.0343527900520848 alphas[1]
   + 0.737946662334014 alphas[2] + 1.461255502958654 alphas[3] - alphas[4]
   + f_var[7] = 0
 R73: - 2419.10426475367 Fraction[7] + Q0_var[7] = 0
 R74: Q0_var[7] + Lost_0_expr_7 = 1167.169859672833
 R75: Q0_var[7] + Q_hat_adjusted_7 = 1447.159403296887
 R76: Sold_0[7] - Q0_var[7] - Q1_var[7] + total_stock_second_period_7 = 0
 R77: total_stock_second_period_7 + Lost_1_expr_7 = 279.9890322391819
 R78: Sold_1[7] + Left_1[7] - total_stock_second_period_7 = 0
 R79: - 700 Sold_0[7] - 700 Sold_1[7] + 700 Lost_0[7] + 700 Lost_1[7]
   + 300 Left_1[7] + profits_vars[7] = 0
 Fraction_lower_bound_8: Fraction[8] >= 0
 Fraction_upper_bound_8: Fraction[8] <= 1
 R82: 0.3492397412676417 alphas[0] + 1.40172235434415 alphas[1]
   - 0.4787072069726699 alphas[2] - 1.123306627911229 alphas[3] - alphas[4]
   + f_var[8] = 0
 R83: - 2419.10426475367 Fraction[8] + Q0_var[8] = 0
 R84: Q0_var[8] + Lost_0_expr_8 = 1946.527279387418
 R85: Q0_var[8] + Q_hat_adjusted_8 = 2411.523072007656
 R86: Sold_0[8] - Q0_var[8] - Q1_var[8] + total_stock_second_period_8 = 0
 R87: total_stock_second_period_8 + Lost_1_expr_8 = 464.995281212029
 R88: Sold_1[8] + Left_1[8] - total_stock_second_period_8 = 0
 R89: - 700 Sold_0[8] - 700 Sold_1[8] + 700 Lost_0[8] + 700 Lost_1[8]
   + 300 Left_1[8] + profits_vars[8] = 0
 Fraction_lower_bound_9: Fraction[9] >= 0
 Fraction_upper_bound_9: Fraction[9] <= 1
 R92: 0.1261854070758301 alphas[0] - 1.4556943350284 alphas[1]
   + 1.357263024233703 alphas[2] - 0.3586881446598509 alphas[3] - alphas[4]
   + f_var[9] = 0
 R93: - 2419.10426475367 Fraction[9] + Q0_var[9] = 0
 R94: Q0_var[9] + Lost_0_expr_9 = 667.2445043369671
 R95: Q0_var[9] + Q_hat_adjusted_9 = 853.0598543528005
 R96: Sold_0[9] - Q0_var[9] - Q1_var[9] + total_stock_second_period_9 = 0
 R97: total_stock_second_period_9 + Lost_1_expr_9 = 185.8148386220886
 R98: Sold_1[9] + Left_1[9] - total_stock_second_period_9 = 0
 R99: - 700 Sold_0[9] - 700 Sold_1[9] + 700 Lost_0[9] + 700 Lost_1[9]
   + 300 Left_1[9] + profits_vars[9] = 0
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
General Constraints
 GC0: Fraction[0] = LOGISTIC ( f_var[0] )
 min_sales_constr_0: Sold_0[0] = MIN ( Q0_var[0] , 1758.349487415308 )
 max_lost_constr_0: Lost_0[0] = MAX ( Lost_0_expr_0 , 0 )
 max_Q1_constr_0: Q1_var[0] = MAX ( Q_hat_adjusted_0 , 0 )
 min_sales2_constr_0: Sold_1[0] = MIN ( total_stock_second_period_0
   , 448.7143865722818 )
 max_lost2_constr_0: Lost_1[0] = MAX ( Lost_1_expr_0 , 0 )
 GC6: Fraction[1] = LOGISTIC ( f_var[1] )
 min_sales_constr_1: Sold_0[1] = MIN ( Q0_var[1] , 1255.034333956949 )
 max_lost_constr_1: Lost_0[1] = MAX ( Lost_0_expr_1 , 0 )
 max_Q1_constr_1: Q1_var[1] = MAX ( Q_hat_adjusted_1 , 0 )
 min_sales2_constr_1: Sold_1[1] = MIN ( total_stock_second_period_1
   , 326.7713413390344 )
 max_lost2_constr_1: Lost_1[1] = MAX ( Lost_1_expr_1 , 0 )
 GC12: Fraction[2] = LOGISTIC ( f_var[2] )
 min_sales_constr_2: Sold_0[2] = MIN ( Q0_var[2] , 2218.733264564339 )
 max_lost_constr_2: Lost_0[2] = MAX ( Lost_0_expr_2 , 0 )
 max_Q1_constr_2: Q1_var[2] = MAX ( Q_hat_adjusted_2 , 0 )
 min_sales2_constr_2: Sold_1[2] = MIN ( total_stock_second_period_2
   , 461.027799735174 )
 max_lost2_constr_2: Lost_1[2] = MAX ( Lost_1_expr_2 , 0 )
 GC18: Fraction[3] = LOGISTIC ( f_var[3] )
 min_sales_constr_3: Sold_0[3] = MIN ( Q0_var[3] , 2075.062390905664 )
 max_lost_constr_3: Lost_0[3] = MAX ( Lost_0_expr_3 , 0 )
 max_Q1_constr_3: Q1_var[3] = MAX ( Q_hat_adjusted_3 , 0 )
 min_sales2_constr_3: Sold_1[3] = MIN ( total_stock_second_period_3
   , 533.1911629643146 )
 max_lost2_constr_3: Lost_1[3] = MAX ( Lost_1_expr_3 , 0 )
 GC24: Fraction[4] = LOGISTIC ( f_var[4] )
 min_sales_constr_4: Sold_0[4] = MIN ( Q0_var[4] , 2413.242143776983 )
 max_lost_constr_4: Lost_0[4] = MAX ( Lost_0_expr_4 , 0 )
 max_Q1_constr_4: Q1_var[4] = MAX ( Q_hat_adjusted_4 , 0 )
 min_sales2_constr_4: Sold_1[4] = MIN ( total_stock_second_period_4
   , 587.7607525014429 )
 max_lost2_constr_4: Lost_1[4] = MAX ( Lost_1_expr_4 , 0 )
 GC30: Fraction[5] = LOGISTIC ( f_var[5] )
 min_sales_constr_5: Sold_0[5] = MIN ( Q0_var[5] , 1971.625493125927 )
 max_lost_constr_5: Lost_0[5] = MAX ( Lost_0_expr_5 , 0 )
 max_Q1_constr_5: Q1_var[5] = MAX ( Q_hat_adjusted_5 , 0 )
 min_sales2_constr_5: Sold_1[5] = MIN ( total_stock_second_period_5
   , 527.9016711963055 )
 max_lost2_constr_5: Lost_1[5] = MAX ( Lost_1_expr_5 , 0 )
 GC36: Fraction[6] = LOGISTIC ( f_var[6] )
 min_sales_constr_6: Sold_0[6] = MIN ( Q0_var[6] , 569.4182127548495 )
 max_lost_constr_6: Lost_0[6] = MAX ( Lost_0_expr_6 , 0 )
 max_Q1_constr_6: Q1_var[6] = MAX ( Q_hat_adjusted_6 , 0 )
 min_sales2_constr_6: Sold_1[6] = MIN ( total_stock_second_period_6
   , 105.8570420498363 )
 max_lost2_constr_6: Lost_1[6] = MAX ( Lost_1_expr_6 , 0 )
 GC42: Fraction[7] = LOGISTIC ( f_var[7] )
 min_sales_constr_7: Sold_0[7] = MIN ( Q0_var[7] , 1167.169859672833 )
 max_lost_constr_7: Lost_0[7] = MAX ( Lost_0_expr_7 , 0 )
 max_Q1_constr_7: Q1_var[7] = MAX ( Q_hat_adjusted_7 , 0 )
 min_sales2_constr_7: Sold_1[7] = MIN ( total_stock_second_period_7
   , 279.9890322391819 )
 max_lost2_constr_7: Lost_1[7] = MAX ( Lost_1_expr_7 , 0 )
 GC48: Fraction[8] = LOGISTIC ( f_var[8] )
 min_sales_constr_8: Sold_0[8] = MIN ( Q0_var[8] , 1946.527279387418 )
 max_lost_constr_8: Lost_0[8] = MAX ( Lost_0_expr_8 , 0 )
 max_Q1_constr_8: Q1_var[8] = MAX ( Q_hat_adjusted_8 , 0 )
 min_sales2_constr_8: Sold_1[8] = MIN ( total_stock_second_period_8
   , 464.995281212029 )
 max_lost2_constr_8: Lost_1[8] = MAX ( Lost_1_expr_8 , 0 )
 GC54: Fraction[9] = LOGISTIC ( f_var[9] )
 min_sales_constr_9: Sold_0[9] = MIN ( Q0_var[9] , 667.2445043369671 )
 max_lost_constr_9: Lost_0[9] = MAX ( Lost_0_expr_9 , 0 )
 max_Q1_constr_9: Q1_var[9] = MAX ( Q_hat_adjusted_9 , 0 )
 min_sales2_constr_9: Sold_1[9] = MIN ( total_stock_second_period_9
   , 185.8148386220886 )
 max_lost2_constr_9: Lost_1[9] = MAX ( Lost_1_expr_9 , 0 )
End
