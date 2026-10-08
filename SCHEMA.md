# SCHEMA (CN synthetic collections data, PS2 files)

All files live in `data/raw/` as CSV. Keys in **bold**.

## accounts.csv (2,400)
**account_id**, lender_id, portfolio {pl_salaried, two_wheeler, mfi_jlg, credit_card,
consumer_durable, msme}, income_type {salaried, self_employed, daily_wage}, town_id {T1,T2,T3},
preferred_language {hinglish, kanglish, english}, bucket_start {X, 1-30, 31-60, 61-90, 90+},
dpd_start, emi_amount, overdue_start, outstanding, salary_credit_day (often null),
bureau_score_band, other_active_loans, paid_other_lenders_30d, last_bounce_reason,
ability_to_pay_estimate (often null), prev_ptp_count, prev_ptp_broken,
dialling_arm {rule_based, random_contact_point}

## splits.csv (2,400)
**account_id**, split {train, validation, test}

## lenders.csv (6)
**lender_id**, lender_name, lender_type, kyc_address_format

## agents.csv (30)
**agent_id** (TA* tele, FA* field, BOT01), channel {tele, field, voice_bot}, language_team,
town_id (field agents), tenure_months, shift

## phones.csv (5,719)
**(account_id, phone_id)** is the unit of prediction. One phone_id can sit on several accounts.
source {kyc_origination, reference, borrower_update, bureau, employer, skip_trace},
relation_recorded {self, reference_spouse, reference_brother, reference_father,
reference_friend, reference_mother, office}, added_date, phone_masked,
priority_slot (0 = dialled first by the rule engine)

## dial_attempts.csv (51,105)
**attempt_id**, account_id, phone_id, attempt_ts, channel {tele_agent, voice_bot}, agent_id,
dialling_arm, selection_propensity (0.25 to 1.0; < 1 only in the random arm),
network_response {answered, ring_no_answer, busy_rejected, not_reachable, switched_off,
number_does_not_exist}, ring_duration_s, talk_duration_s, hangup_by {network, customer, agent},
disposition {rpc_ptp, rpc_hung_up, rpc_call_back, rpc_refused, rpc_hardship, rpc_dispute,
rpc_claims_paid, third_party_contact, third_party_ptp, wrong_number, no_answer,
call_rejected, not_reachable, switched_off, invalid_number, language_barrier},
remark (free text, 3 languages, typos), ptp_id, has_transcript

Disposition is typed by the agent or bot and can be wrong.

## payments.csv (2,162)
**payment_id**, account_id, payment_ts, amount, channel

## skip_traces.csv (766)
**trace_id**, account_id, trace_date, trigger_rule (always 15_consecutive_failed_contacts),
result {no_new_info, new_phone_found, new_address_found}, new_contact_point_id, cost_inr

## verified_contact_points.csv (250), GOLD LABELS
**(account_id, phone_id)**, verified_status {borrower_number, third_party_number,
not_borrower_number, switched_off, invalid_number}, verified_date (2026-07-02)
Read only via `rpc.io.load_gold(splits=...)`.

## addresses.csv (3,117)
**address_id**, account_id, address_type {residence, office, permanent_native},
source {kyc_origination, skip_trace}, added_date, town_id {T1,T2,T3,OUT}, address_text

## field_visits.csv (5,578)
**visit_id**, account_id, address_id, agent_id, visit_date, start_ts, checkin_ts,
checkin_x, checkin_y, gps_accuracy_m, dwell_s, outcome {met_borrower, met_family,
locked_premises, address_not_traceable, neighbour_says_shifted, no_such_person,
cash_collected}, ptp_id, remark, photo_hash
