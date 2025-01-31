-- слегка замудренный запрос
-- сперва отбираем читателей и книги, что они брали
-- с доп. информацией о том, какую книгу они взяли в прошлый, а какую в следующий раз
-- , если таковые есть. в обратном случае будет пустое значение
with cte_gr as (
	select 
		reader_id
		, book_id
		, borrow_date
		, lag(borrow_date) over (partition by reader_id order by borrow_date) as previous_borrow_time
		, lag(expected_return_date) over (partition by reader_id order by borrow_date) as previous_return_time
		, lead(borrow_date) over (partition by reader_id order by borrow_date) as next_borrow_time
		, lead(expected_return_date) over (partition by reader_id order by borrow_date) as next_return_time
	from borrowed_books
)
select 
	concat(r.name, ' ', r.surname, ' ', r.patronym) as full_name
	, b.title
	, a.author_name
	, cte.previous_borrow_time
	, cte.previous_return_time
	, cte.next_borrow_time
	, cte.next_return_time
from readers r
join cte_gr cte using(reader_id)
join books b using(book_id)
join authors a on b.author_id = a.author_id
join genres g on g.genre_id = b.genre_id


-- Типовые запросы

-- Список книг с просроченным возвратом
SELECT 
	b.title
	, r.name
	, bb.expected_return_date
	, bb.actual_return_date
FROM books b
JOIN borrowed_books bb 
	USING(book_id)
JOIN readers r
	USING(reader_id)
WHERE bb.actual_return_date > bb.expected_return_date;

-- Список читателей с просроченным возвратом
SELECT 
	r.name
	, r.phone_number
	, bb.expected_return_date
	, bb.actual_return_date
FROM readers r
JOIN borrowed_books bb 
	USING(reader_id)
WHERE bb.actual_return_date > bb.expected_return_date;

-- Общая стоимость штрафов по каждому читателю
SELECT 
	r.name
	, SUM(f.amount) AS total_fines
FROM readers r
JOIN borrowed_books bb 
	USING(reader_id)
JOIN fines f ON bb.borrow_id = f.borrow_id
GROUP BY r.name;

-- Уведомления, которые еще не отправлены
SELECT 
	rn.notification_text
	, rn.send_date
	, r.name
FROM reader_notifications rn
JOIN readers r ON 
	USING(reader_id)
WHERE rn.status = FALSE;


