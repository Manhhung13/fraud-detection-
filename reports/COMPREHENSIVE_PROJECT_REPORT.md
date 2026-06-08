# Báo Cáo Tổng Hợp Dự Án Phát Hiện Gian Lận (Fraud Detection)

**Mục tiêu dự án:** Xây dựng mô hình Machine Learning phát hiện giao dịch gian lận trong dữ liệu tài chính. 
**Thách thức chính:** Dữ liệu mất cân bằng nhãn trầm trọng (chỉ 2.55% là gian lận). Do đó, dự án không dùng Accuracy mà ưu tiên tối ưu **PR-AUC, F2-score, Precision, Recall** và **Review Rate**.

---

## 1. Khảo Sát Dữ Liệu (Data Understanding & EDA)

Tập dữ liệu bao gồm **100,000 giao dịch** với 36 đặc trưng (features).

**Đặc điểm nhãn (Target Distribution):**
- Giao dịch bình thường (0): 97,451 (97.45%)
- Giao dịch gian lận (1): **2,549 (2.55%)**

**Nhận xét từ EDA:**
- Dữ liệu có nhiều biến phân loại (categorical) với độ đa dạng (cardinality) rất cao (ví dụ: `mcc`, `merchant_id`, `ip_risk`). Các biến này không thể dùng phương pháp One-Hot Encoding trực tiếp mà cần gom nhóm (rare grouping) hoặc tính toán các biến tương tác.
- Các feature thời gian (`local_timestamp`) cần tách thành giờ, ngày trong tuần, và phân loại giao dịch ban đêm/ngày.
- Nhóm biến thể hiện **lịch sử giao dịch của người dùng** (trong 7, 30 ngày) đóng vai trò then chốt (như `mcc_entropy_30d`, `max_amount_30d`, `night_ratio_30d`).

---

## 2. Khai Phá Dữ Liệu Ẩn (Insight Mining)

Dự án áp dụng phương pháp phân tích Subgroup Discovery để tìm ra các quy luật có nguy cơ gian lận cao gấp nhiều lần so với mức trung bình (Lift > 2.0).

**Các hành vi rủi ro cao (Behavioral Anomalies):**
1. **Ít đa dạng ngành hàng (Low MCC Entropy):** Khách hàng bình thường có xu hướng chi tiêu ở nhiều ngành hàng. Khi lịch sử chi tiêu ít đa dạng (`mcc_entropy_30d` thấp) nhưng bỗng phát sinh giao dịch khác lạ, rủi ro gian lận tăng gấp **3.7 lần**.
2. **Giao dịch đêm bất thường (Night Unusual):** Nếu khách hàng ít khi giao dịch ban đêm (`night_ratio_30d` thấp) nhưng lại xuất hiện giao dịch vào thời gian này, tỷ lệ gian lận tăng gấp **2.8 lần**.
3. **Bất thường số tiền (Amount Anomaly):** Các giao dịch có số tiền lệch chuẩn lớn so với lịch sử (`max_amount_30d`) có rủi ro cao gấp **2.4 lần**.
4. **Ít đa dạng quốc gia (Low Country Diversity):** Tương tự như ngành hàng, người dùng ít giao dịch xuyên quốc gia nhưng có giao dịch quốc tế bất ngờ sẽ có rủi ro lớn hơn.

**Quy tắc kết hợp (Interaction Rules):**
Khi các tín hiệu bất thường kết hợp với nhau (ví dụ: `low_mcc_entropy` + `low_night_ratio` + `card_not_present`), tỷ lệ gian lận có thể vọt lên tới **26.7%** (tăng gấp 10 lần so với trung bình).

---

## 3. Trích Xuất Đặc Trưng (Feature Engineering)

Từ các insight trên, pipeline xử lý đã tạo ra **115 features** để phục vụ mô hình hoá, bao gồm:
- **Numeric Features:** Điểm bất thường thời gian (`night_unusual_score`), tỷ lệ giao dịch đêm (`night_ratio_30d`), độ đa dạng thiết bị/quốc gia, và các đặc trưng chuẩn hóa số tiền (`amount_z_30d`).
- **Binary/Interaction Features:** Các cờ nhị phân như `low_mcc_entropy_flag`, `card_not_present_flag` và biến kết hợp `low_mcc_entropy_x_low_night_ratio`.
- **Ghi chú quan trọng (Data Leakage):** Tất cả các thông số lịch sử đều được tính toán **trước** thời điểm diễn ra giao dịch để tránh việc mô hình nhìn thấy tương lai.

---

## 4. Huấn Luyện & Đánh Giá Mô Hình (Model Training & Selection)

Dự án đã thực hiện kiểm thử trên các thuật toán: **Logistic Regression** (Baseline), Random Forest, LightGBM, XGBoost và **CatBoost** (Advanced).

**1. Baseline Model (Logistic Regression):**
- Được dùng làm cơ sở diễn giải nhờ tính minh bạch của các hệ số (coefficients).
- Đạt PR-AUC: 0.5332, Precision: 40.4%, Recall: 69.8% (Ngưỡng 0.77).
- Phân tích hệ số của Logistic Regression chứng minh rằng mô hình học đúng các insight đã đào sâu (tăng risk cho các biến dị thường về thời gian, số tiền và giảm risk cho độ đa dạng hành vi).

**2. Final Model (CatBoost - Được lựa chọn):**
- **Cấu hình:** `cat_tune_base_depth6_l2_5_lr003_iter700` (Sử dụng MCC rare grouping, không gán sẵn interaction manual).
- **Lý do chọn:** CatBoost thể hiện sức mạnh vượt trội khi tự học các tương tác phi tuyến mà không cần tạo bằng tay, cho ra PR-AUC và F2 score tốt nhất.
- **Top Feature Importance:** Khớp hoàn toàn với insight ban đầu: `mcc_entropy_30d` (Top 1), `night_ratio_30d` (Top 2), `log_amount_to_max_30d` (Top 3).

---

## 5. Phân Tích Hiệu Năng & Lỗi (Performance & Error Analysis)

**Trên tập Test (15,000 giao dịch - 388 Fraud):**
Mô hình CatBoost (với ngưỡng quyết định `0.74`) đem lại kết quả:
- **True Positives (TP):** 289 (Bắt đúng gian lận)
- **False Positives (FP):** 260 (Báo động nhầm)
- **False Negatives (FN):** 99 (Bỏ sót gian lận)
- **True Negatives (TN):** 14,352 (Xác định đúng giao dịch thường)

**Metrics đạt được:**
- **Recall:** 74.48% (Tóm được gần 3/4 lượng giao dịch lừa đảo)
- **Precision:** 52.64% (Cứ 2 cảnh báo thì có 1 cảnh báo là đúng)
- **Review Rate:** 3.66% (Đội vận hành chỉ phải kiểm tra thủ công 3.66% trên tổng lưu lượng).

**Phân tích lỗi (Error Analysis):**
- Lỗi bỏ sót (FN) thường tập trung vào các giao dịch có độ đa dạng thiết bị/quốc gia cao, số tiền thấp, hoặc dùng thiết bị như Smart Watch.
- Lỗi bắt nhầm (FP) thường xảy ra ở kênh IVR_PHONE, các giao dịch có mã vào thẻ manual.

---

## 6. Chính Sách Nghiệp Vụ (Threshold Policy)

Dựa vào năng lực kiểm duyệt (Review Capacity) của hệ thống, dự án đề xuất các mức kiểm duyệt linh hoạt:

1. **Năng lực Review 3%:** Ngưỡng threshold `0.73` -> Bắt được `72.1%` gian lận.
2. **Năng lực Review 4%:** Ngưỡng threshold `0.74` -> Bắt được `74.5%` gian lận (Khuyên dùng).
3. **Năng lực Review 5%:** Ngưỡng threshold `0.66` -> Bắt được `77.5%` gian lận.

**Hệ thống phân tầng rủi ro (Risk Tiers):**
- `Score < 0.3`: **LOW RISK** (Tự động duyệt - Approve)
- `0.3 <= Score < 0.74`: **MEDIUM RISK** (Đưa vào danh sách xem xét hậu kiểm hoặc xác thực OTP phụ)
- `Score >= 0.74`: **HIGH RISK** (Chặn tự động hoặc yêu cầu Review ngay lập tức)

---
