-- Таблица филиалов
CREATE TABLE branches (
    branch_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    city VARCHAR(100) NOT NULL,
    address TEXT NOT NULL
);

-- добавить таблицу с дополнительными книгами на продажу
-- плюсы: 
-- привлечем новых читателей
-- увеличим врение к чтению более сложной литературы у школьников, через продажу им легкой литературы


-- Таблица сотрудников
CREATE TABLE employees (
    employee_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    position VARCHAR(100) NOT NULL,
    hire_date DATE NOT NULL,
    termination_date DATE,
    passport_data VARCHAR(50) NOT NULL,
    phone_number VARCHAR(15),
    branch_id INT NOT NULL REFERENCES branches(branch_id)
);

-- после создания таблицы с сотрудниками
-- можем привязать id менеджера к id сотрудника
ALTER TABLE branches
ADD COLUMN manager_id INT NULL REFERENCES employees(employee_id)

-- Таблица жанров
CREATE TABLE genres (
    genre_id SERIAL PRIMARY KEY,         
    genre_name VARCHAR(100) NOT NULL    
);
-- Таблица авторов
CREATE TABLE authors (
    author_id SERIAL PRIMARY KEY,       
    author_name VARCHAR(255) NOT NULL   
);

-- Таблица книг
CREATE TABLE books (
    book_id SERIAL PRIMARY KEY,             
    title VARCHAR(255) NOT NULL,            
    author_id INT NOT NULL REFERENCES authors(author_id) ON DELETE CASCADE, 
    genre_id INT REFERENCES genres(genre_id), 
    publisher VARCHAR(255),                 
    publication_year INT,                   
    shelf_location VARCHAR(50),             
    available BOOLEAN NOT NULL DEFAULT TRUE,
    branch_id INT NOT NULL REFERENCES branches(branch_id) 
);

-- Таблица читателей
CREATE TABLE readers (
    reader_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
	surname VARCHAR(255) NOT NULL,
	patronym VARCHAR(255) NOT NULL,
    phone_number VARCHAR(15),
    passport_data VARCHAR(50) NOT NULL,
    registered_branch_id INT NOT NULL REFERENCES branches(branch_id)
);

-- Таблица взятых книг
CREATE TABLE borrowed_books (
    borrow_id SERIAL PRIMARY KEY,
    reader_id INT NOT NULL REFERENCES readers(reader_id),
    book_id INT NOT NULL REFERENCES books(book_id),
    branch_id INT NOT NULL REFERENCES branches(branch_id),
    borrow_date DATE NOT NULL,
    expected_return_date DATE NOT NULL,
    actual_return_date DATE 
);
-- добавляем ограничения на даты
ALTER TABLE borrowed_books
ADD CONSTRAINT check_actual_dates CHECK (actual_return_date IS NULL OR actual_return_date >= borrow_date);

ALTER TABLE borrowed_books
ADD CONSTRAINT check_expected_dates CHECK (expected_return_date IS NULL OR expected_return_date >= borrow_date);


-- Таблица штрафов
CREATE TABLE fines (
    fine_id SERIAL PRIMARY KEY,
    borrow_id INT NOT NULL REFERENCES borrowed_books(borrow_id),
    amount NUMERIC(10, 2) NOT NULL,
    payment_status BOOLEAN NOT NULL DEFAULT FALSE
);

-- в случае, если будет реализована идея 
-- с книжным магазином при библиотеке, то могут пригодиться следующие таблицы:
-- Таблица поставщиков
-- CREATE TABLE suppliers (
--     supplier_id SERIAL PRIMARY KEY,
--     name VARCHAR(255) NOT NULL,
--     contact_person VARCHAR(255),
--     phone_number VARCHAR(15),
--     email VARCHAR(100),
--     address TEXT
-- );

-- Таблица поставок книг
-- CREATE TABLE book_deliveries (
--     delivery_id SERIAL PRIMARY KEY,
--     supplier_id INT NOT NULL REFERENCES suppliers(supplier_id),
--     branch_id INT NOT NULL REFERENCES branches(branch_id),
--     delivery_date DATE NOT NULL,
--     status VARCHAR(50) NOT NULL,
--     total_books INT NOT NULL
-- );

-- Таблица элементов поставки
-- CREATE TABLE delivery_items (
--     item_id SERIAL PRIMARY KEY,
--     delivery_id INT NOT NULL REFERENCES book_deliveries(delivery_id),
--     book_id INT NOT NULL REFERENCES books(book_id),
--     quantity INT NOT NULL
-- );

-- Таблица уведомлений
CREATE TABLE reader_notifications (
    notification_id SERIAL PRIMARY KEY,
    reader_id INT NOT NULL REFERENCES readers(reader_id),
    notification_text TEXT NOT NULL,
    send_date DATE NOT NULL,
    status BOOLEAN NOT NULL DEFAULT FALSE
);

-- Триггеры и функции

-- Функция для начисления штрафа при просрочке возврата книги
CREATE OR REPLACE FUNCTION calculate_fine() RETURNS TRIGGER AS $$
BEGIN
    IF NEW.actual_return_date > NEW.expected_return_date THEN
        INSERT INTO fines (borrow_id, amount, payment_status)
        VALUES (NEW.borrow_id, 
                EXTRACT(WEEK FROM (NEW.actual_return_date - NEW.expected_return_date)) * 10,
                FALSE);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Триггер для вызова функции начисления штрафа
CREATE TRIGGER trigger_calculate_fine
AFTER UPDATE OF actual_return_date ON borrowed_books
FOR EACH ROW
EXECUTE FUNCTION calculate_fine();


-- Функция для создания уведомления о приближающемся сроке возврата книги
CREATE OR REPLACE FUNCTION notify_reader() RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO reader_notifications (reader_id, notification_text, send_date, status)
    VALUES (NEW.reader_id, 
            'Срок возврата книги приближается. Пожалуйста, верните книгу или продлите срок.',
            CURRENT_DATE, 
            FALSE);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Триггер для уведомлений читателям
CREATE TRIGGER trigger_notify_reader
BEFORE UPDATE OF expected_return_date ON borrowed_books
FOR EACH ROW
WHEN (NEW.expected_return_date - INTERVAL '5 days' <= CURRENT_DATE)
EXECUTE FUNCTION notify_reader(); -- если еще не была возвращена

-- тригер на случай, если не указан филиал библиотеки 
-- при вставки значения в таблицу с заимствованиями
CREATE OR REPLACE FUNCTION set_default_branch() RETURNS TRIGGER AS $$
BEGIN
    -- Устанавливаем branch_id из таблицы books
    NEW.branch_id := (
        SELECT branch_id
        FROM books
        WHERE book_id = NEW.book_id
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_set_branch
BEFORE INSERT ON borrowed_books
FOR EACH ROW
WHEN (NEW.branch_id IS NULL) -- Срабатывает только если branch_id не задан
EXECUTE FUNCTION set_default_branch();


-- Процедуры

-- Процедура для обработки выдачи книги
CREATE OR REPLACE PROCEDURE issue_book(reader INT, book INT, borrow_date DATE, return_date DATE)
LANGUAGE plpgsql
AS $$
BEGIN
    -- Проверяем, доступна ли книга
    IF NOT EXISTS (SELECT 1 FROM books WHERE book_id = book AND available = TRUE) THEN
        RAISE EXCEPTION 'Книга недоступна для выдачи';
    END IF;

    -- Добавляем запись о выдаче книги
    INSERT INTO borrowed_books (reader_id, book_id, borrow_date, expected_return_date)
    VALUES (reader, book, borrow_date, return_date);

    -- Обновляем статус книги на недоступный
    UPDATE books SET available = FALSE WHERE book_id = book;
END;
$$;

-- Процедура для обработки возврата книги
CREATE OR REPLACE PROCEDURE return_book(reader INT, book INT, return_date DATE)
LANGUAGE plpgsql
AS $$
BEGIN
     -- Проверяем, действительно ли читатель брал эту книгу
    IF NOT EXISTS (
        SELECT 1 
        FROM borrowed_books
        WHERE 1=1
			AND reader_id = reader 
			AND book_id = book 
			AND actual_return_date IS NULL
    ) THEN
        RAISE EXCEPTION 'Читатель с id % не брал книгу с id % или она уже возвращена', reader, book;
    END IF;

    UPDATE borrowed_books
    SET actual_return_date = return_date
    WHERE 1=1
		AND reader_id = reader
		AND book_id = book;

    UPDATE books
    SET available = TRUE
    WHERE book_id = book;
END;
$$;

-- Представления (VIEW)

-- 1. Книги с просроченным временем сдачи
CREATE OR REPLACE VIEW overdue_books AS
SELECT 
	b.book_id
	, b.title
	, r.reader_id
	, r.name AS reader_name
	, bb.expected_return_date
	, bb.actual_return_date
FROM books b
JOIN borrowed_books bb 
	USING(book_id)
JOIN readers r 
	USING(reader_id)
WHERE 
	bb.actual_return_date > bb.expected_return_date OR 
	(bb.actual_return_date IS NULL AND bb.expected_return_date < CURRENT_DATE);

-- 2. Читатели, что просрочили дату возврата книги
CREATE OR REPLACE VIEW overdue_readers AS
SELECT 
	r.reader_id
	, r.name AS reader_name
	, r.phone_number
	, bb.expected_return_date
	, bb.actual_return_date
FROM readers r
JOIN borrowed_books bb 
	USING(reader_id)
WHERE 
	bb.actual_return_date > bb.expected_return_date OR 
	(bb.actual_return_date IS NULL AND bb.expected_return_date < CURRENT_DATE);
