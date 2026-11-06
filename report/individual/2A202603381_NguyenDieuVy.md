# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603381
- **Họ và tên:** Nguyễn Diệu Vy
- **SEED** (in ở ô đầu notebook): 1008

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

                                     f1  f1_SUB    p95_ms  usd_per_1k
1. BoW + LogReg (tự code)        0.905   0.919     0.118       0.000
2. BoW + MLP (deep learning)     0.960   0.950     0.373       0.000
3a. Encoder (MiniLM) + kNN       0.925   1.000   118.783       0.000
3b. Encoder (MiniLM) + LogReg    0.957   0.974    35.947       0.000
3c. Encoder (MiniLM) + MLP head  0.952   0.974    35.457       0.000
4a. LLM nhỏ zero-shot            0.923   0.923  1351.673       0.006
4b. LLM lớn zero-shot            0.974   0.974  1331.859       0.063
4c. LLM nhỏ few-shot             0.974   0.974  1761.343       0.016

## 3. Dự đoán và thực tế

1. **Bỏ ReLU (2.B).** Mình đoán F1 giảm về ngang logistic regression (khoảng 0.905). Thực tế mạng không ReLU đạt 0.955, gần bằng MLP có ReLU (0.960). Mình lệch vì nghĩ mức tăng của MLP đến từ phi tuyến. Mạng không ReLU vẫn là mô hình tuyến tính, nên chênh lệch với Phần 1 nhiều khả năng đến từ cách huấn luyện: Adam, mini-batch 64 và 30 epoch (khoảng 1.700 bước cập nhật) so với gradient descent toàn bộ dữ liệu 300 bước ở Phần 1, nơi loss vẫn còn đang giảm. Mình chưa kiểm chứng trực tiếp điều này.
2. **Tin ngụy trang (3.B).** Mình đoán Encoder + LogReg đúng nhiều hơn BoW. Thực tế BoW đúng 5/6, Encoder + LogReg đúng 4/6. Cả hai cùng sai tin phishing "Last chance! Your account rewards expire tonight...", và embedding sai thêm tin "complimentary holiday". Mình lệch vì tin rằng mô hình hiểu nghĩa sẽ tự khắc tốt hơn, trong khi đầu phân loại chỉ học từ các ví dụ spam có trong tập train, và 6 tin là mẫu quá nhỏ.
3. **Chi phí few-shot (4.B).** Mình đoán chi phí mỗi tin tăng 4 đến 6 lần và F1 gần như không đổi. Thực tế chi phí tăng khoảng 2.5 lần ($0.016 so với $0.006 mỗi 1k tin) và F1 tăng từ 0.923 lên 0.974 (khoảng 2 tin trên 40). Mình lệch vì ước lượng sai độ dài của bốn ví dụ so với prompt gốc.

## 4. Phân tích 50 case khó

Accuracy trên 50 case: BoW + LogReg 54%, BoW + MLP 56%, 3a 72%, 3b 66%, 3c 68%, LLM nhỏ 100% (cả prompt thường và prompt phòng thủ). Tụt nhiều nhất là hai mô hình BoW: F1 từ 0.919 xuống 0.378 (BoW + LogReg) và từ 0.950 xuống 0.421 (BoW + MLP).

| Nhóm | BoW+LR | BoW+MLP | 3a kNN | 3b | 3c | LLM |
| --- | --- | --- | --- | --- | --- | --- |
| Tưởng spam nhưng không phải | 100% | 90% | 70% | 80% | 70% | 100% |
| Thao túng / lừa đảo | 12% | 38% | 100% | 50% | 50% | 100% |
| Spam giọng tự nhiên | 20% | 10% | 40% | 20% | 50% | 100% |
| Né từ khoá | 17% | 0% | 83% | 100% | 83% | 100% |
| Prompt injection | 67% | 83% | 83% | 100% | 100% | 100% |
| Nói về spam | 100% | 100% | 25% | 50% | 50% | 100% |
| Tin giao dịch hợp lệ | 83% | 100% | 100% | 83% | 83% | 100% |

- **Né từ khoá** (ví dụ #29 "C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd..."): BoW chỉ đúng 1/6 (BoW + MLP 0/6). Thay chữ bằng số tạo ra các token như `c0ngr4tulat10ns` không có trong từ điển 2000 từ nên bị bỏ qua, mất hết tín hiệu spam. Embedding đúng 83 đến 100% vì encoder vẫn nhận ra nghĩa của câu.
- **Thao túng / lừa đảo** (ví dụ #21 "Hi Mum, I dropped my phone in the toilet... send £200"): BoW chỉ đúng 1/8. Tin không chứa từ khoá spam, cái bẫy nằm ở ý định (mạo danh người thân, xin tiền, tạo cảm giác gấp) mà việc cộng trọng số từng từ không nắm được. Ngược lại, nhóm "Nói về spam" thì embedding yếu (kNN chỉ đúng 25%) còn BoW đúng 100%, nhiều khả năng vì embedding gần với chủ đề spam hơn là ý định của người nói.
- LLM đúng cả 50 case, kể cả nhóm prompt injection, nên mô hình này không bị lừa bởi "Ignore all previous instructions" và việc so sánh prompt phòng thủ với prompt thường không phân biệt được gì (cả hai đều 100%).

## 5. Thử thách cá nhân

**Đề #1 (ML):** đổi ngưỡng quyết định 0.5 sang 0.2 và 0.8 cho mô hình Phần 1 (SEED 1008, 1008 mod 6 = 0). Mình tính precision, recall, F1, FP, FN trên tập test:

| Ngưỡng | Precision | Recall | F1 | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0.2 | 0.932 | 0.923 | 0.928 | 7 | 8 |
| 0.5 | 1.000 | 0.827 | 0.905 | 0 | 18 |
| 0.8 | 1.000 | 0.663 | 0.798 | 0 | 35 |

Hạ ngưỡng thì chặn dễ hơn nên recall tăng nhưng chặn nhầm tin ham (precision giảm); nâng ngưỡng thì chỉ chặn khi rất chắc nên lọt nhiều spam hơn. Ngân hàng sợ chặn nhầm tin OTP nên chọn **0.5**: ở 0.2 có 7 tin ham bị chặn nhầm, còn 0.5 và 0.8 đều có FP = 0 nhưng 0.5 giữ recall cao hơn (lọt 18 thay vì 35 spam). Nên kết hợp danh sách trắng cho OTP, vì FP = 0 trên tập test chưa chắc đúng ngoài thực tế.

## 6. Phương án cho sản phẩm thật

Yêu cầu: p95 < 50 ms/tin cho 10 triệu SMS/ngày (khoảng 116 tin/giây).
- **Loại LLM làm bộ lọc duy nhất:** p95 khoảng 1,3 đến 1,8 giây, không đạt SLA. Chi phí theo bảng của mình: LLM nhỏ $0.006/1k tin, tức khoảng $62/ngày (khoảng $1.900/tháng), LLM lớn $0.063/1k tin, khoảng $630/ngày (khoảng $19.000/tháng).
- **Các cách đạt SLA:** BoW + LogReg (0.12 ms), BoW + MLP (0.37 ms), Encoder + LogReg (35.95 ms), Encoder + MLP head (35.46 ms). BoW + MLP có F1 test cao nhất (0.960) nhưng mong manh nhất: chỉ đúng 56% trên 50 case khó. Encoder + MLP head đạt F1 test 0.952 và 68% trên case khó, p95 35.46 ms.
- **Chọn thiết kế lai:** BoW + MLP chấm mọi tin trước (rất nhanh, rẻ); tin có xác suất lưỡng lự chuyển sang Encoder + MLP head; chỉ các tin vẫn lưỡng lự mới gửi LLM nhỏ. Giả sử 5% tin tới LLM thì chi phí LLM khoảng $3/ngày (con số này là giả định, cần đo tỉ lệ thật).
- **Đánh đổi:** hệ thống phức tạp hơn; Encoder chỉ đạt khoảng 35 tin/giây trong lần đo của mình nên cần nhiều tiến trình song song hoặc GPU để chịu 116 tin/giây; các tin tới LLM vẫn mất khoảng 1 giây; và LLM không nên là lớp kiểm tra duy nhất vì rủi ro prompt injection.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| Cấu hình API | Đã thêm key nhưng notebook chưa nhận, sửa thế nào? | Hướng dẫn đọc Colab Secrets và gán biến môi trường | Chạy lại ô cấu hình, in ra provider và tên model, xác nhận HAS_LLM = True và gọi thử API thật được |
| Phần 4: Prompt và parse | Hoàn thành hàm tạo prompt và đọc nhãn | Viết prompt zero-shot/few-shot và parse_label | Chạy ô kiểm tra 4.2 đến 4.3; thử hai tin thật, nhận nhãn [1, 0], invalid = 0 |
| Phần 4.A: Zero-shot | Viết đầy đủ code so sánh hai model | Thêm khoảng nghỉ, xử lý rate limit và đặt giới hạn 30 token | Lần chạy đầu cả hai model F1 = 0 và invalid = 40 (model trả về rỗng); sau khi sửa giới hạn token thì invalid = 0, model nhỏ F1 = 0.947, model lớn F1 = 0.919. Chênh 0.028 tương đương khoảng 1 tin trên 40 tin SUB nên coi là hòa, không kết luận model lớn kém hơn |
| Phần 4.B: Few-shot | Viết code chạy few-shot và so với zero-shot | Thêm bốn ví dụ từ train, tính F1 và chi phí | F1 = 0.974 so với 0.947 zero-shot (chênh khoảng 1 tin, chưa đủ để kết luận few-shot tốt hơn); chi phí mỗi tin tăng 2.25 lần, thấp hơn dự đoán 4 đến 6 lần của mình, invalid = 0 |
| Stress test | Viết code đánh giá các model trên case khó | Viết vòng lặp đánh giá và bảng kết quả theo nhóm | Chạy và đối chiếu bảng % đúng theo nhóm với dự đoán của mình; đọc danh sách case mà tất cả model đều sai |

**Điều mình hiểu ra hôm nay:** mức tăng F1 từ logistic regression lên MLP (0.905 lên 0.960) hóa ra gần bằng mức tăng của mạng tuyến tính không ReLU (0.955), nên một phần lớn đến từ cách huấn luyện chứ không hẳn từ lớp ẩn. Điểm cao trên tập test thường không nói lên khả năng chống lại tin cố tình đánh lừa: hai mô hình BoW từ F1 khoảng 0.92 đến 0.95 tụt xuống 0.38 đến 0.42 trên 50 case khó. Và mọi lựa chọn phải đo theo ràng buộc thật (độ trễ, chi phí, loại lỗi đắt hơn), không chỉ theo F1.
