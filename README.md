> **Lời giải** nằm ở [`olpai-factcheck/`](olpai-factcheck/) (CV 5-fold 0.7103; xem README và EXPERIMENTS.md trong đó).
> Phần dưới là mô tả dữ liệu gốc của Ban tổ chức.

# Bài 1: Kiểm chứng tin tức (dữ liệu)

Các tệp nằm cùng thư mục với notebook. Đề bài đầy đủ nằm trong `statement.md`.

| Tệp | Nội dung |
|---|---|
| `statement.md` | Đề bài, định dạng nộp và cách chấm |
| `train.csv` | 4.434 nhận định có nhãn (`id`, `article_id`, `claim`, `evidence`, `label`) |
| `articles.csv` | Bài báo của tập train và **public test** (`article_id`, `source`, `topic`, `text`) |
| `public_test.csv` | 817 nhận định public test (`id`, `article_id`, `claim`) |
| `sample_submission.csv` | Bài nộp mẫu cho public test, hai cột `id`, `label` |
| `sample_submission.zip` | Cùng nội dung, đã nén đúng định dạng nộp |
| `private_test.zip` | Tập private test, **có mật khẩu**. Ban tổ chức công bố mật khẩu trong giờ thi private |
| `notebook_mau.ipynb` | Notebook mẫu và baseline |
| `scorer.py` | Trình chấm chạy trên máy của đội thi |

## Hai giai đoạn

1. **Public test** (hầu hết thời gian): dự đoán `public_test.csv`. Bài báo tương ứng đã có trong `articles.csv`.
2. **Private test** (khoảng một giờ cuối): giải nén `private_test.zip` **vào thư mục này** (cùng chỗ với notebook), rồi chạy lại notebook.

```
unzip private_test.zip
```

Sau khi giải nén sẽ có thêm `private_test.csv` và `private_articles.csv`. Notebook tự chuyển sang giai đoạn private nếu thấy hai tệp đó.

## Nộp bài

Tệp nộp là một tệp **zip chứa đúng một tệp CSV**. Tệp CSV có hai cột `id` và `label`, đủ mọi `id` của tập test **đang chấm** (`public_test.csv` hoặc `private_test.csv`). Notebook mẫu ghi `bai1_submission.csv` rồi nén thành `submission.zip`.

Dữ liệu lấy từ bộ ViFactCheck (Hoa và cộng sự, AAAI 2025), giấy phép MIT. Ban tổ chức đã chuẩn hoá văn bản và chia lại dữ liệu theo bài báo.
