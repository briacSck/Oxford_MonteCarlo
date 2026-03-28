// Analysis for the paper "Mapping Entrepreneurial Inclusion Across US Neighborhoods: The Case of Low-Code E-commerce Entrepreneurship"
// published in the Strategic Management Journal
// Bryan Stroube (bstroube@london.edu) and Gary Dushnitsky (gdushnitsky@london.edu)

version 18 // Run on Stata version 18
clear

use mapping_entrepreneurial_inclusion.dta

// Table 1: Summary statistics.
estpost summarize shopify_count pop_black_aa pop_total total_bachelor_deg pop_total_poverty total_social_cap all_new_bus vcbacked_nonshopify_count
eststo table1

// Table 2: Correlation table.
pwcorr shopify_count pop_black_aa pop_total total_bachelor_deg pop_total_poverty total_social_cap all_new_bus vcbacked_nonshopify_count

// Table 3: Main results predicting the likelihood of Shopify ventures within ZCTAs.
qui reg log_shopify_count_1 log_pop_black_aa log_pop_total log_total_bachelor_deg log_pop_total_poverty log_total_social_cap i.state_name_fe i.MSA_fe, vce(cluster MSA_fe)
eststo table3_1

qui reg shopify_count_nonzero log_pop_black_aa log_pop_total log_total_bachelor_deg log_pop_total_poverty log_total_social_cap i.state_name_fe i.MSA_fe, vce(cluster MSA_fe)
eststo table3_2

qui reg log_shopify_count_1 log_pop_black_aa log_pop_total log_total_bachelor_deg log_pop_total_poverty log_total_social_cap i.state_name_fe i.MSA_fe if shopify_count > 0, vce(cluster MSA_fe)
eststo table3_3

// Table 3 formatted output
estout table3_1 table3_2 table3_3, label unstack cells(b (fmt (4)) p) stats(r2 N) drop(*.state_name_fe *.MSA_fe)


// Table 4: "Book ended" benchmark analyses.
qui reg log_all_new_bus log_pop_black_aa log_pop_total log_total_bachelor_deg log_pop_total_poverty log_total_social_cap i.state_name_fe i.MSA_fe, vce(cluster MSA_fe)
eststo table4_1

qui reg log_vcbacked_nonshopify_count_1 log_pop_black_aa log_pop_total log_total_bachelor_deg log_pop_total_poverty log_total_social_cap i.state_name_fe i.MSA_fe, vce(cluster MSA_fe)
eststo table4_2

// Table 4 formatted output
estout table4_1 table4_2, label unstack cells(b (fmt (4)) p) stats(r2 N) drop(*.state_name_fe *.MSA_fe)
