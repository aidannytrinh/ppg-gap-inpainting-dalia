# Điền khuyết đoạn PPG mất liên tục 2 giây trên PPG-DaLiA

Mô hình tự mã hóa **1D-CNN kết hợp Bi-LSTM** (117.233 tham số) khôi phục 2 giây tín hiệu PPG (BVP) bị mất, dùng ngữ cảnh cả hai phía của đoạn mất. Tiểu luận cuối khóa môn *Trí tuệ nhân tạo cho IoT*, Khoa Công nghệ Thông tin, Trường Đại học Công nghệ Kỹ thuật TP.HCM.

- Sinh viên: Trịnh Nhật Anh, MSSV 23110074 (đề tài G6)
- Giảng viên hướng dẫn: ThS. Hồ Nhựt Minh
- Lớp học phần: 261AIOT331185_01CLC, học kỳ I, năm học 2026 - 2027
- Bộ dữ liệu: PPG-DaLiA, 15 người đeo vòng Empatica E4 trong sinh hoạt hằng ngày (UCI, DOI 10.24432/C53890)

## Bài toán

Trong mỗi cửa sổ 8 giây (512 mẫu ở 64 Hz), che đi 128 mẫu liên tục (2 giây). Mô hình nhận tín hiệu đã bị che cùng mặt nạ cho biết chỗ bị che, rồi dự đoán lại 128 mẫu đó. Phần bị che chỉ dùng để chấm điểm, không bao giờ được đưa cho mô hình.

So sánh với ba phương pháp đối chuẩn: nội suy tuyến tính, spline bậc ba và lặp chu kỳ tim gần nhất.

## Kết quả chính

Đánh giá trên 3 người test (S13 đến S15), tập này bị khóa trong mã và chỉ được mở sau khi đã khóa cấu hình.

| Nội dung | Mô hình | So sánh |
|---|---|---|
| Sai số đoạn bị che (MAE, gộp theo đối tượng) | 0,5079 | lặp chu kỳ 0,9712 (tốt nhất trong ba đối chuẩn) |
| Lệch nhịp tim so với PPG đầy đủ | 2,99 bpm | lặp chu kỳ 5,15 bpm |
| Lệch nhịp tim so với ECG | 6,61 bpm | bộ đọc trên PPG đầy đủ đã lệch 6,99 bpm |
| Kiểm định chéo 5 fold (15 người) | 0,5500 ± 0,0303 | lặp chu kỳ 0,9893 ± 0,0207, mô hình tốt hơn ở 15/15 người |
| Bỏ tầng Bi-LSTM | MAE tăng lên 0,6808 | kém hơn ở 3/3 người test |
| Thời gian tính mỗi cửa sổ trên CPU | 9,34 ms | mục tiêu dưới 100 ms |
| Thời gian chờ dữ liệu phía sau đoạn mất | 3 giây | cải tiến còn 1 giây, MAE chỉ tăng 3,5% |

### Giới hạn

- Mất 2 giây liên tục là kịch bản mô phỏng, không phải kiểu mất dữ liệu đo từ thiết bị thật.
- Bộ dữ liệu không có PPG sạch tuyệt đối, nên đích để so là tín hiệu đã ghi, không phải dạng sóng sinh lý thật.
- Độ trễ chỉ đo trên máy tính (CPU), chưa thử trên thiết bị đeo.
- **Không đạt** mục tiêu sai số nhịp tim so với ECG dưới 5 bpm (đạt 6,61 bpm). Bộ đọc nhịp tim trên chính PPG đầy đủ cũng đã lệch 6,99 bpm, nên phần lớn sai số này đến từ bộ đọc, không chỉ từ việc điền khuyết.
- Phần khám phá dữ liệu (`kham_pha_du_lieu.py`) được vẽ sau khi đã có kết quả, dùng để kiểm tra lại các quyết định thiết kế, không làm thay đổi kết quả nào.

## Cấu trúc thư mục

```
ppg-gap-inpainting-dalia/
  README.md
  data/PPG_FieldStudy/   bộ dữ liệu gốc (tự tải về, không nằm trong repo)
  dalia_cache/           cache nhỏ do tao_cache.py tạo ra (không nằm trong repo)
  G6/                    mã nguồn, kết quả, trọng số
```

Trong `G6/`:

| Tệp | Vai trò |
|---|---|
| `tao_cache.py` | Trích BVP, ACC cổ tay và nhãn hoạt động từ `SX.pkl` gốc ra `dalia_cache` |
| `prep_hr.py` | Trích nhãn HR tham chiếu từ ECG, kiểm tra khớp số cửa sổ |
| `nguon_du_lieu.py` | Ghi nguồn, phiên bản, mã SHA-256 của dữ liệu |
| `danh_sach_split_fold.py` | Ghi danh sách đối tượng theo split cố định và theo 5 fold |
| `data.py` | Nạp dữ liệu, cắt cửa sổ, mặt nạ, chuẩn hóa theo phần quan sát, chia tập và 5 fold |
| `kham_pha_du_lieu.py` | Khám phá dữ liệu, chỉ dùng S1 đến S12 (không đọc test) |
| `hr_reader.py` | Bộ đọc nhịp tim cố định (Butterworth 0,5 đến 8 Hz và tìm đỉnh), chọn tham số trên train/validation |
| `baselines.py` | Ba phương pháp đối chuẩn |
| `model.py` | Kiến trúc 1D-CNN + Bi-LSTM |
| `losses.py` | Hàm mất mát trên khoảng thiếu và số hạng sai phân tại biên |
| `train.py` | Vòng huấn luyện, dừng sớm, checkpoint từng epoch để chạy tiếp khi bị ngắt |
| `evaluate.py` | MAE, RMSE, lệch HR so với PPG đầy đủ và so với ECG, gộp theo đối tượng, đánh giá theo hoạt động |
| `experiments.py` | Điều phối E4, E2, khóa cấu hình, E1 theo đúng thứ tự |
| `chay_e5.py` | Chạy trọn E5 (kiểm định chéo, 20 lượt huấn luyện) rồi đánh giá ngoài mẫu |
| `e2_du_luoi.py` | Chọn λ của hai bản bóc tách E2 trên đủ lưới {0; 0,1; 0,5; 1} |
| `e4_noi_mep.py` | Sai số nối mép của các mô hình E4 trên validation |
| `khao_sat_vi_tri.py` | Độ nhạy theo vị trí khoảng thiếu {64, 128, 192, 256, 320} trên test |
| `cai_tien_do_tre.py` | Cải tiến (ngoài đề cương): đo sai số theo thời gian chờ sau đoạn mất, chọn mức chờ chỉ bằng validation |
| `phan_tich_lech_hr.py` | Chiều lệch HR của từng phương pháp so với PPG đầy đủ |
| `bench.py` | Đo tài nguyên và độ trễ |
| `figures.py`, `ve_so_do_khoi.py`, `xuat_bang.py` | Sinh hình và bảng từ tệp kết quả |
| `tests_leak.py` | 29 kiểm thử bắt buộc và kiểm thử chống rò rỉ |
| `G6_Colab.ipynb` | Chạy cùng các bước trên Google Colab |
| `requirements.txt` | Phiên bản thư viện |

Thư mục kết quả: `results/` (json, csv), `weights/` (33 trọng số `.pt`), `figs/` (hình), `logs/` (nhật ký chạy).

## Cài đặt

```
pip install -r G6/requirements.txt
```

## Chuẩn bị dữ liệu

1. Tải PPG-DaLiA từ UCI Machine Learning Repository (DOI 10.24432/C53890), giải nén để có `data/PPG_FieldStudy/S1 ... S15` ở thư mục gốc của repo.
2. Tạo cache và nhãn HR:

```
cd G6
python tao_cache.py
python prep_hr.py
python nguon_du_lieu.py
python danh_sach_split_fold.py
```

`nguon_du_lieu.py` ghi mã SHA-256 của từng tệp vào `results/nguon_du_lieu.json`, dùng để đối chiếu đúng phiên bản dữ liệu.

## Chạy lại toàn bộ theo đúng giao thức

Chạy trong thư mục `G6`, theo đúng thứ tự. Tập test S13 đến S15 bị khóa trong mã, chỉ mở sau bước khóa cấu hình.

```
python tests_leak.py                          # kiểm thử bắt buộc, phải đạt hết
python data.py --thongke                      # thống kê cửa sổ theo đối tượng
python hr_reader.py --chon                    # chọn tham số bộ đọc HR trên S1-S12
python evaluate.py --doichuan --tap val       # ba đối chuẩn trên validation

python experiments.py e4                      # 4 lượt, lambda trong {0; 0,1; 0,5; 1}
python experiments.py e2 --luoi 0 0.1 0.5 1   # bóc tách: bỏ Bi-LSTM, thêm ACC, đủ 4 giá trị λ
python experiments.py khoa                    # khóa cấu hình, ghi thời điểm
python experiments.py e1                      # so sánh và phân tích theo hoạt động trên test S13-S15

python chay_e5.py                             # E5: 20 lượt rồi đánh giá ngoài mẫu

python e2_du_luoi.py                          # đối chiếu lựa chọn λ của E2 trên đủ lưới với file khóa
python e4_noi_mep.py                          # nối mép E4, chỉ dùng validation
python kham_pha_du_lieu.py                    # khám phá dữ liệu, chỉ dùng S1-S12
python khao_sat_vi_tri.py --cho_phep_test     # 5 vị trí khoảng thiếu, báo cáo tách riêng
python khao_sat_vi_tri.py --gop
python phan_tich_lech_hr.py

# Cải tiến ngoài đề cương: giảm độ trễ chờ ngữ cảnh (quy tắc chốt trước trong results/caitien_*_quy_tac.json)
python cai_tien_do_tre.py --tap val
python cai_tien_do_tre.py --tap test --cho_phep_test
python train.py --id caitien_amax384_lam0.5 --lam 0.5 --a_max 384 --a_eval 320 352 384
python cai_tien_do_tre.py --tap val --mo_hinh_them caitien_amax384_lam0.5
python cai_tien_do_tre.py --tap test --cho_phep_test --mo_hinh_them caitien_amax384_lam0.5
python cai_tien_do_tre.py --gop

python bench.py                               # chạy khi máy rảnh, không huấn luyện song song
python figures.py
python ve_so_do_khoi.py
python xuat_bang.py
```

Mỗi lượt huấn luyện bỏ qua nếu đã có `results/<id>.json`, và tự chạy tiếp từ checkpoint nếu bị ngắt giữa chừng, nên chạy lại cùng lệnh là an toàn.

Trọng số đã huấn luyện có sẵn trong `G6/weights/` (mô hình đã chọn và các biến thể E2, E4, E5), nên không bắt buộc huấn luyện lại để đánh giá, nhưng vẫn cần `dalia_cache`.

## Những gì không đưa lên repo

Để repo gọn (khoảng 17 MB), các tệp sau không có trong repo mà được tạo lại khi chạy mã:

- Bộ dữ liệu gốc `data/PPG_FieldStudy` và `dalia_cache` (tải về và tạo bằng `tao_cache.py`).
- Tệp dự đoán theo từng cửa sổ `results/*_theo_cuaso.csv` và `results/*_dudoan_gap.npz` (khoảng 230 MB), do `evaluate.py` và `experiments.py e1` sinh ra. `figures.py`, `xuat_bang.py` và `phan_tich_lech_hr.py` cần các tệp này, nên hãy chạy bước E1 trước.
- Thư mục `results/truoc_bo_sung_e2/` (kết quả trước khi bổ sung E2, xem ghi chú bên dưới).

## Ghi chú về cách tạo kết quả và notebook

- **Cách tạo kết quả:** các kết quả gốc được tạo bằng cách chạy trực tiếp các script `.py` theo thứ tự ở mục "Chạy lại toàn bộ theo đúng giao thức". Ba lượt khảo sát λ = 0; 0,1; 0,5 chạy trên Colab (GPU Tesla T4), các lượt còn lại chạy trên CPU máy cá nhân. Thiết bị và phiên bản thư viện được ghi trong từng tệp `results/<mã lượt>.json`. Các tệp trong `results/`, `logs/` và `figs/` chính là kết quả gốc đó.
- **Notebook:** `G6/G6_Colab.ipynb` chỉ gọi lần lượt các script theo thứ tự chính bằng lệnh `!python ...` để tiện chạy trên Colab. Notebook **không chứa output** và chưa gồm các bước bổ sung (khám phá dữ liệu, khảo sát vị trí, phân tích lệch HR, cải tiến độ trễ). Danh sách lệnh đầy đủ nằm ở README này. Notebook không phải nơi lưu kết quả.
- **Tên thí nghiệm:** báo cáo gọi các thí nghiệm bằng tên nội dung, còn đề cương và tên tệp dùng mã. Đối chiếu: E4 là khảo sát λ, E2 là bóc tách, E1 là so sánh trên test, E3 là phân tích theo hoạt động, E5 là kiểm định chéo.

## Thời gian chạy tham khảo

Trên CPU Intel Core i7-12700H, một epoch mất khoảng 20 đến 30 giây khi máy rảnh. Mỗi lượt dừng sớm sau khoảng 70 đến 100 epoch, tức khoảng 25 đến 50 phút. Trên Colab Tesla T4 khoảng 49 đến 52 giây mỗi epoch. Toàn bộ thí nghiệm theo đề cương gồm 32 lượt huấn luyện: E4 có 4 lượt, E2 có 8 lượt, E5 có 20 lượt. Phần cải tiến có thêm 1 lượt là biến thể độ trễ thấp `caitien_amax384_lam0.5`.

## Ghi chú trung thực về quy trình

- **E2 chạy bổ sung sau khi khóa cấu hình.** Lượt đầu chỉ thử λ ∈ {0; 0,5} rồi mới khóa cấu hình. Bốn lượt còn thiếu (λ = 0,1 và 1) được chạy bổ sung sau thời điểm khóa, theo quy tắc ghi trước trong `results/e2_bo_sung_quy_tac.json` (chọn MAE validation nhỏ nhất, không dùng test). Cả hai bản đổi sang λ = 1; lựa chọn mới nằm trong `results/khoa_cau_hinh_bo_sung_e2.json`, file khóa gốc giữ nguyên. Nếu chạy lại từ đầu theo thứ tự trên, bước khóa sẽ chọn thẳng λ = 1 cho hai bản này.
- **Môi trường chạy thực tế.** Ba lượt E4 với lambda = 0; 0,1; 0,5 chạy trên Google Colab Tesla T4, các lượt còn lại chạy trên CPU máy cá nhân. Mỗi `results/<id>.json` ghi rõ thiết bị, phiên bản thư viện và thời gian chạy trong mục `moi_truong` và `thoi_gian`.
- Seed chính là 42. Kết quả trên CPU và GPU có thể lệch rất nhỏ ở chữ số cuối do cách tính dấu phẩy động khác nhau, không ảnh hưởng tới kết luận.
- Tệp `logs/*.log` ghi đường dẫn tuyệt đối của máy đã chạy, giữ nguyên làm nhật ký.

## Trích dẫn bộ dữ liệu

Reiss, A., Indlekofer, I., Schmidt, P., Van Laerhoven, K. (2019). *Deep PPG: Large-scale heart rate estimation with convolutional neural networks.* Sensors, 19(14), 3079. Bộ dữ liệu: PPG-DaLiA, UCI Machine Learning Repository, DOI 10.24432/C53890.
