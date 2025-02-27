PGDMP  5                    |            library_management    16.6 (Postgres.app)    16.3 X    �           0    0    ENCODING    ENCODING        SET client_encoding = 'UTF8';
                      false            �           0    0 
   STDSTRINGS 
   STDSTRINGS     (   SET standard_conforming_strings = 'on';
                      false            �           0    0 
   SEARCHPATH 
   SEARCHPATH     8   SELECT pg_catalog.set_config('search_path', '', false);
                      false            �           1262    16968    library_management    DATABASE     �   CREATE DATABASE library_management WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = icu LOCALE = 'en_US.UTF-8' ICU_LOCALE = 'en-US';
 "   DROP DATABASE library_management;
                postgres    false            �            1255    17534    calculate_fine()    FUNCTION     �  CREATE FUNCTION public.calculate_fine() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.actual_return_date > NEW.expected_return_date THEN
        INSERT INTO fines (borrow_id, amount, payment_status)
        VALUES (NEW.borrow_id, 
                EXTRACT(WEEK FROM (NEW.actual_return_date - NEW.expected_return_date)) * 10,
                FALSE);
    END IF;
    RETURN NEW;
END;
$$;
 '   DROP FUNCTION public.calculate_fine();
       public          postgres    false            �            1255    17555 (   issue_book(integer, integer, date, date) 	   PROCEDURE     �  CREATE PROCEDURE public.issue_book(IN reader integer, IN book integer, IN borrow_date date, IN return_date date)
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
 p   DROP PROCEDURE public.issue_book(IN reader integer, IN book integer, IN borrow_date date, IN return_date date);
       public          postgres    false            �            1255    17536    notify_reader()    FUNCTION     �  CREATE FUNCTION public.notify_reader() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    INSERT INTO reader_notifications (reader_id, notification_text, send_date, status)
    VALUES (NEW.reader_id, 
            'Срок возврата книги приближается. Пожалуйста, верните книгу или продлите срок.',
            CURRENT_DATE, 
            FALSE);
    RETURN NEW;
END;
$$;
 &   DROP FUNCTION public.notify_reader();
       public          postgres    false            �            1255    17563 #   return_book(integer, integer, date) 	   PROCEDURE       CREATE PROCEDURE public.return_book(IN reader integer, IN book integer, IN return_date date)
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
 \   DROP PROCEDURE public.return_book(IN reader integer, IN book integer, IN return_date date);
       public          postgres    false            �            1255    17557    set_default_branch()    FUNCTION     .  CREATE FUNCTION public.set_default_branch() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    -- Устанавливаем branch_id из таблицы books
    NEW.branch_id := (
        SELECT branch_id
        FROM books
        WHERE book_id = NEW.book_id
    );
    RETURN NEW;
END;
$$;
 +   DROP FUNCTION public.set_default_branch();
       public          postgres    false            �            1259    17438    authors    TABLE     q   CREATE TABLE public.authors (
    author_id integer NOT NULL,
    author_name character varying(255) NOT NULL
);
    DROP TABLE public.authors;
       public         heap    postgres    false            �            1259    17437    authors_author_id_seq    SEQUENCE     �   CREATE SEQUENCE public.authors_author_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 ,   DROP SEQUENCE public.authors_author_id_seq;
       public          postgres    false    222            �           0    0    authors_author_id_seq    SEQUENCE OWNED BY     O   ALTER SEQUENCE public.authors_author_id_seq OWNED BY public.authors.author_id;
          public          postgres    false    221            �            1259    17445    books    TABLE     X  CREATE TABLE public.books (
    book_id integer NOT NULL,
    title character varying(255) NOT NULL,
    author_id integer NOT NULL,
    genre_id integer,
    publisher character varying(255),
    publication_year integer,
    shelf_location character varying(50),
    available boolean DEFAULT true NOT NULL,
    branch_id integer NOT NULL
);
    DROP TABLE public.books;
       public         heap    postgres    false            �            1259    17444    books_book_id_seq    SEQUENCE     �   CREATE SEQUENCE public.books_book_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 (   DROP SEQUENCE public.books_book_id_seq;
       public          postgres    false    224            �           0    0    books_book_id_seq    SEQUENCE OWNED BY     G   ALTER SEQUENCE public.books_book_id_seq OWNED BY public.books.book_id;
          public          postgres    false    223            �            1259    17485    borrowed_books    TABLE     �  CREATE TABLE public.borrowed_books (
    borrow_id integer NOT NULL,
    reader_id integer NOT NULL,
    book_id integer NOT NULL,
    branch_id integer NOT NULL,
    borrow_date date NOT NULL,
    expected_return_date date NOT NULL,
    actual_return_date date,
    CONSTRAINT check_actual_dates CHECK (((actual_return_date IS NULL) OR (actual_return_date >= borrow_date))),
    CONSTRAINT check_expected_dates CHECK (((expected_return_date IS NULL) OR (expected_return_date >= borrow_date)))
);
 "   DROP TABLE public.borrowed_books;
       public         heap    postgres    false            �            1259    17484    borrowed_books_borrow_id_seq    SEQUENCE     �   CREATE SEQUENCE public.borrowed_books_borrow_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 3   DROP SEQUENCE public.borrowed_books_borrow_id_seq;
       public          postgres    false    228            �           0    0    borrowed_books_borrow_id_seq    SEQUENCE OWNED BY     ]   ALTER SEQUENCE public.borrowed_books_borrow_id_seq OWNED BY public.borrowed_books.borrow_id;
          public          postgres    false    227            �            1259    17405    branches    TABLE     �   CREATE TABLE public.branches (
    branch_id integer NOT NULL,
    name character varying(255) NOT NULL,
    city character varying(100) NOT NULL,
    address text NOT NULL,
    manager_id integer
);
    DROP TABLE public.branches;
       public         heap    postgres    false            �            1259    17404    branches_branch_id_seq    SEQUENCE     �   CREATE SEQUENCE public.branches_branch_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 -   DROP SEQUENCE public.branches_branch_id_seq;
       public          postgres    false    216            �           0    0    branches_branch_id_seq    SEQUENCE OWNED BY     Q   ALTER SEQUENCE public.branches_branch_id_seq OWNED BY public.branches.branch_id;
          public          postgres    false    215            �            1259    17414 	   employees    TABLE     P  CREATE TABLE public.employees (
    employee_id integer NOT NULL,
    name character varying(255) NOT NULL,
    "position" character varying(100) NOT NULL,
    hire_date date NOT NULL,
    termination_date date,
    passport_data character varying(50) NOT NULL,
    phone_number character varying(15),
    branch_id integer NOT NULL
);
    DROP TABLE public.employees;
       public         heap    postgres    false            �            1259    17413    employees_employee_id_seq    SEQUENCE     �   CREATE SEQUENCE public.employees_employee_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 0   DROP SEQUENCE public.employees_employee_id_seq;
       public          postgres    false    218            �           0    0    employees_employee_id_seq    SEQUENCE OWNED BY     W   ALTER SEQUENCE public.employees_employee_id_seq OWNED BY public.employees.employee_id;
          public          postgres    false    217            �            1259    17507    fines    TABLE     �   CREATE TABLE public.fines (
    fine_id integer NOT NULL,
    borrow_id integer NOT NULL,
    amount numeric(10,2) NOT NULL,
    payment_status boolean DEFAULT false NOT NULL
);
    DROP TABLE public.fines;
       public         heap    postgres    false            �            1259    17506    fines_fine_id_seq    SEQUENCE     �   CREATE SEQUENCE public.fines_fine_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 (   DROP SEQUENCE public.fines_fine_id_seq;
       public          postgres    false    230            �           0    0    fines_fine_id_seq    SEQUENCE OWNED BY     G   ALTER SEQUENCE public.fines_fine_id_seq OWNED BY public.fines.fine_id;
          public          postgres    false    229            �            1259    17431    genres    TABLE     n   CREATE TABLE public.genres (
    genre_id integer NOT NULL,
    genre_name character varying(100) NOT NULL
);
    DROP TABLE public.genres;
       public         heap    postgres    false            �            1259    17430    genres_genre_id_seq    SEQUENCE     �   CREATE SEQUENCE public.genres_genre_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 *   DROP SEQUENCE public.genres_genre_id_seq;
       public          postgres    false    220            �           0    0    genres_genre_id_seq    SEQUENCE OWNED BY     K   ALTER SEQUENCE public.genres_genre_id_seq OWNED BY public.genres.genre_id;
          public          postgres    false    219            �            1259    17470    readers    TABLE     J  CREATE TABLE public.readers (
    reader_id integer NOT NULL,
    name character varying(255) NOT NULL,
    surname character varying(255) NOT NULL,
    patronym character varying(255) NOT NULL,
    phone_number character varying(15),
    passport_data character varying(50) NOT NULL,
    registered_branch_id integer NOT NULL
);
    DROP TABLE public.readers;
       public         heap    postgres    false            �            1259    17540    overdue_books    VIEW     �  CREATE VIEW public.overdue_books AS
 SELECT b.book_id,
    b.title,
    r.reader_id,
    r.name AS reader_name,
    bb.expected_return_date,
    bb.actual_return_date
   FROM ((public.books b
     JOIN public.borrowed_books bb USING (book_id))
     JOIN public.readers r USING (reader_id))
  WHERE ((bb.actual_return_date > bb.expected_return_date) OR ((bb.actual_return_date IS NULL) AND (bb.expected_return_date < CURRENT_DATE)));
     DROP VIEW public.overdue_books;
       public          postgres    false    228    224    228    224    226    228    226    228            �            1259    17545    overdue_readers    VIEW     �  CREATE VIEW public.overdue_readers AS
 SELECT r.reader_id,
    r.name AS reader_name,
    r.phone_number,
    bb.expected_return_date,
    bb.actual_return_date
   FROM (public.readers r
     JOIN public.borrowed_books bb USING (reader_id))
  WHERE ((bb.actual_return_date > bb.expected_return_date) OR ((bb.actual_return_date IS NULL) AND (bb.expected_return_date < CURRENT_DATE)));
 "   DROP VIEW public.overdue_readers;
       public          postgres    false    226    228    228    226    226    228            �            1259    17520    reader_notifications    TABLE     �   CREATE TABLE public.reader_notifications (
    notification_id integer NOT NULL,
    reader_id integer NOT NULL,
    notification_text text NOT NULL,
    send_date date NOT NULL,
    status boolean DEFAULT false NOT NULL
);
 (   DROP TABLE public.reader_notifications;
       public         heap    postgres    false            �            1259    17519 (   reader_notifications_notification_id_seq    SEQUENCE     �   CREATE SEQUENCE public.reader_notifications_notification_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 ?   DROP SEQUENCE public.reader_notifications_notification_id_seq;
       public          postgres    false    232            �           0    0 (   reader_notifications_notification_id_seq    SEQUENCE OWNED BY     u   ALTER SEQUENCE public.reader_notifications_notification_id_seq OWNED BY public.reader_notifications.notification_id;
          public          postgres    false    231            �            1259    17469    readers_reader_id_seq    SEQUENCE     �   CREATE SEQUENCE public.readers_reader_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 ,   DROP SEQUENCE public.readers_reader_id_seq;
       public          postgres    false    226            �           0    0    readers_reader_id_seq    SEQUENCE OWNED BY     O   ALTER SEQUENCE public.readers_reader_id_seq OWNED BY public.readers.reader_id;
          public          postgres    false    225            �           2604    17441    authors author_id    DEFAULT     v   ALTER TABLE ONLY public.authors ALTER COLUMN author_id SET DEFAULT nextval('public.authors_author_id_seq'::regclass);
 @   ALTER TABLE public.authors ALTER COLUMN author_id DROP DEFAULT;
       public          postgres    false    222    221    222            �           2604    17448    books book_id    DEFAULT     n   ALTER TABLE ONLY public.books ALTER COLUMN book_id SET DEFAULT nextval('public.books_book_id_seq'::regclass);
 <   ALTER TABLE public.books ALTER COLUMN book_id DROP DEFAULT;
       public          postgres    false    224    223    224            �           2604    17488    borrowed_books borrow_id    DEFAULT     �   ALTER TABLE ONLY public.borrowed_books ALTER COLUMN borrow_id SET DEFAULT nextval('public.borrowed_books_borrow_id_seq'::regclass);
 G   ALTER TABLE public.borrowed_books ALTER COLUMN borrow_id DROP DEFAULT;
       public          postgres    false    228    227    228            �           2604    17408    branches branch_id    DEFAULT     x   ALTER TABLE ONLY public.branches ALTER COLUMN branch_id SET DEFAULT nextval('public.branches_branch_id_seq'::regclass);
 A   ALTER TABLE public.branches ALTER COLUMN branch_id DROP DEFAULT;
       public          postgres    false    216    215    216            �           2604    17417    employees employee_id    DEFAULT     ~   ALTER TABLE ONLY public.employees ALTER COLUMN employee_id SET DEFAULT nextval('public.employees_employee_id_seq'::regclass);
 D   ALTER TABLE public.employees ALTER COLUMN employee_id DROP DEFAULT;
       public          postgres    false    217    218    218            �           2604    17510    fines fine_id    DEFAULT     n   ALTER TABLE ONLY public.fines ALTER COLUMN fine_id SET DEFAULT nextval('public.fines_fine_id_seq'::regclass);
 <   ALTER TABLE public.fines ALTER COLUMN fine_id DROP DEFAULT;
       public          postgres    false    230    229    230            �           2604    17434    genres genre_id    DEFAULT     r   ALTER TABLE ONLY public.genres ALTER COLUMN genre_id SET DEFAULT nextval('public.genres_genre_id_seq'::regclass);
 >   ALTER TABLE public.genres ALTER COLUMN genre_id DROP DEFAULT;
       public          postgres    false    220    219    220            �           2604    17523 $   reader_notifications notification_id    DEFAULT     �   ALTER TABLE ONLY public.reader_notifications ALTER COLUMN notification_id SET DEFAULT nextval('public.reader_notifications_notification_id_seq'::regclass);
 S   ALTER TABLE public.reader_notifications ALTER COLUMN notification_id DROP DEFAULT;
       public          postgres    false    231    232    232            �           2604    17473    readers reader_id    DEFAULT     v   ALTER TABLE ONLY public.readers ALTER COLUMN reader_id SET DEFAULT nextval('public.readers_reader_id_seq'::regclass);
 @   ALTER TABLE public.readers ALTER COLUMN reader_id DROP DEFAULT;
       public          postgres    false    225    226    226            �          0    17438    authors 
   TABLE DATA           9   COPY public.authors (author_id, author_name) FROM stdin;
    public          postgres    false    222   wy       �          0    17445    books 
   TABLE DATA           �   COPY public.books (book_id, title, author_id, genre_id, publisher, publication_year, shelf_location, available, branch_id) FROM stdin;
    public          postgres    false    224   Gz       �          0    17485    borrowed_books 
   TABLE DATA           �   COPY public.borrowed_books (borrow_id, reader_id, book_id, branch_id, borrow_date, expected_return_date, actual_return_date) FROM stdin;
    public          postgres    false    228   �~       ~          0    17405    branches 
   TABLE DATA           N   COPY public.branches (branch_id, name, city, address, manager_id) FROM stdin;
    public          postgres    false    216   N       �          0    17414 	   employees 
   TABLE DATA           �   COPY public.employees (employee_id, name, "position", hire_date, termination_date, passport_data, phone_number, branch_id) FROM stdin;
    public          postgres    false    218   =�       �          0    17507    fines 
   TABLE DATA           K   COPY public.fines (fine_id, borrow_id, amount, payment_status) FROM stdin;
    public          postgres    false    230   ��       �          0    17431    genres 
   TABLE DATA           6   COPY public.genres (genre_id, genre_name) FROM stdin;
    public          postgres    false    220   �       �          0    17520    reader_notifications 
   TABLE DATA           p   COPY public.reader_notifications (notification_id, reader_id, notification_text, send_date, status) FROM stdin;
    public          postgres    false    232   f�       �          0    17470    readers 
   TABLE DATA           x   COPY public.readers (reader_id, name, surname, patronym, phone_number, passport_data, registered_branch_id) FROM stdin;
    public          postgres    false    226   ��       �           0    0    authors_author_id_seq    SEQUENCE SET     C   SELECT pg_catalog.setval('public.authors_author_id_seq', 9, true);
          public          postgres    false    221            �           0    0    books_book_id_seq    SEQUENCE SET     A   SELECT pg_catalog.setval('public.books_book_id_seq', 179, true);
          public          postgres    false    223            �           0    0    borrowed_books_borrow_id_seq    SEQUENCE SET     K   SELECT pg_catalog.setval('public.borrowed_books_borrow_id_seq', 11, true);
          public          postgres    false    227            �           0    0    branches_branch_id_seq    SEQUENCE SET     D   SELECT pg_catalog.setval('public.branches_branch_id_seq', 5, true);
          public          postgres    false    215            �           0    0    employees_employee_id_seq    SEQUENCE SET     H   SELECT pg_catalog.setval('public.employees_employee_id_seq', 29, true);
          public          postgres    false    217            �           0    0    fines_fine_id_seq    SEQUENCE SET     @   SELECT pg_catalog.setval('public.fines_fine_id_seq', 1, false);
          public          postgres    false    229            �           0    0    genres_genre_id_seq    SEQUENCE SET     A   SELECT pg_catalog.setval('public.genres_genre_id_seq', 3, true);
          public          postgres    false    219            �           0    0 (   reader_notifications_notification_id_seq    SEQUENCE SET     W   SELECT pg_catalog.setval('public.reader_notifications_notification_id_seq', 1, false);
          public          postgres    false    231            �           0    0    readers_reader_id_seq    SEQUENCE SET     D   SELECT pg_catalog.setval('public.readers_reader_id_seq', 28, true);
          public          postgres    false    225            �           2606    17443    authors authors_pkey 
   CONSTRAINT     Y   ALTER TABLE ONLY public.authors
    ADD CONSTRAINT authors_pkey PRIMARY KEY (author_id);
 >   ALTER TABLE ONLY public.authors DROP CONSTRAINT authors_pkey;
       public            postgres    false    222            �           2606    17453    books books_pkey 
   CONSTRAINT     S   ALTER TABLE ONLY public.books
    ADD CONSTRAINT books_pkey PRIMARY KEY (book_id);
 :   ALTER TABLE ONLY public.books DROP CONSTRAINT books_pkey;
       public            postgres    false    224            �           2606    17490 "   borrowed_books borrowed_books_pkey 
   CONSTRAINT     g   ALTER TABLE ONLY public.borrowed_books
    ADD CONSTRAINT borrowed_books_pkey PRIMARY KEY (borrow_id);
 L   ALTER TABLE ONLY public.borrowed_books DROP CONSTRAINT borrowed_books_pkey;
       public            postgres    false    228            �           2606    17412    branches branches_pkey 
   CONSTRAINT     [   ALTER TABLE ONLY public.branches
    ADD CONSTRAINT branches_pkey PRIMARY KEY (branch_id);
 @   ALTER TABLE ONLY public.branches DROP CONSTRAINT branches_pkey;
       public            postgres    false    216            �           2606    17419    employees employees_pkey 
   CONSTRAINT     _   ALTER TABLE ONLY public.employees
    ADD CONSTRAINT employees_pkey PRIMARY KEY (employee_id);
 B   ALTER TABLE ONLY public.employees DROP CONSTRAINT employees_pkey;
       public            postgres    false    218            �           2606    17513    fines fines_pkey 
   CONSTRAINT     S   ALTER TABLE ONLY public.fines
    ADD CONSTRAINT fines_pkey PRIMARY KEY (fine_id);
 :   ALTER TABLE ONLY public.fines DROP CONSTRAINT fines_pkey;
       public            postgres    false    230            �           2606    17436    genres genres_pkey 
   CONSTRAINT     V   ALTER TABLE ONLY public.genres
    ADD CONSTRAINT genres_pkey PRIMARY KEY (genre_id);
 <   ALTER TABLE ONLY public.genres DROP CONSTRAINT genres_pkey;
       public            postgres    false    220            �           2606    17528 .   reader_notifications reader_notifications_pkey 
   CONSTRAINT     y   ALTER TABLE ONLY public.reader_notifications
    ADD CONSTRAINT reader_notifications_pkey PRIMARY KEY (notification_id);
 X   ALTER TABLE ONLY public.reader_notifications DROP CONSTRAINT reader_notifications_pkey;
       public            postgres    false    232            �           2606    17477    readers readers_pkey 
   CONSTRAINT     Y   ALTER TABLE ONLY public.readers
    ADD CONSTRAINT readers_pkey PRIMARY KEY (reader_id);
 >   ALTER TABLE ONLY public.readers DROP CONSTRAINT readers_pkey;
       public            postgres    false    226            �           2620    17535 %   borrowed_books trigger_calculate_fine    TRIGGER     �   CREATE TRIGGER trigger_calculate_fine AFTER UPDATE OF actual_return_date ON public.borrowed_books FOR EACH ROW EXECUTE FUNCTION public.calculate_fine();
 >   DROP TRIGGER trigger_calculate_fine ON public.borrowed_books;
       public          postgres    false    228    235    228            �           2620    17537 $   borrowed_books trigger_notify_reader    TRIGGER     �   CREATE TRIGGER trigger_notify_reader BEFORE UPDATE OF expected_return_date ON public.borrowed_books FOR EACH ROW WHEN (((new.expected_return_date - '5 days'::interval) <= CURRENT_DATE)) EXECUTE FUNCTION public.notify_reader();
 =   DROP TRIGGER trigger_notify_reader ON public.borrowed_books;
       public          postgres    false    228    228    228    236            �           2620    17558 !   borrowed_books trigger_set_branch    TRIGGER     �   CREATE TRIGGER trigger_set_branch BEFORE INSERT ON public.borrowed_books FOR EACH ROW WHEN ((new.branch_id IS NULL)) EXECUTE FUNCTION public.set_default_branch();
 :   DROP TRIGGER trigger_set_branch ON public.borrowed_books;
       public          postgres    false    238    228    228            �           2606    17454    books books_author_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.books
    ADD CONSTRAINT books_author_id_fkey FOREIGN KEY (author_id) REFERENCES public.authors(author_id) ON DELETE CASCADE;
 D   ALTER TABLE ONLY public.books DROP CONSTRAINT books_author_id_fkey;
       public          postgres    false    3539    222    224            �           2606    17464    books books_branch_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.books
    ADD CONSTRAINT books_branch_id_fkey FOREIGN KEY (branch_id) REFERENCES public.branches(branch_id);
 D   ALTER TABLE ONLY public.books DROP CONSTRAINT books_branch_id_fkey;
       public          postgres    false    216    224    3533            �           2606    17459    books books_genre_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.books
    ADD CONSTRAINT books_genre_id_fkey FOREIGN KEY (genre_id) REFERENCES public.genres(genre_id);
 C   ALTER TABLE ONLY public.books DROP CONSTRAINT books_genre_id_fkey;
       public          postgres    false    3537    224    220            �           2606    17496 *   borrowed_books borrowed_books_book_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.borrowed_books
    ADD CONSTRAINT borrowed_books_book_id_fkey FOREIGN KEY (book_id) REFERENCES public.books(book_id);
 T   ALTER TABLE ONLY public.borrowed_books DROP CONSTRAINT borrowed_books_book_id_fkey;
       public          postgres    false    228    224    3541            �           2606    17501 ,   borrowed_books borrowed_books_branch_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.borrowed_books
    ADD CONSTRAINT borrowed_books_branch_id_fkey FOREIGN KEY (branch_id) REFERENCES public.branches(branch_id);
 V   ALTER TABLE ONLY public.borrowed_books DROP CONSTRAINT borrowed_books_branch_id_fkey;
       public          postgres    false    216    3533    228            �           2606    17491 ,   borrowed_books borrowed_books_reader_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.borrowed_books
    ADD CONSTRAINT borrowed_books_reader_id_fkey FOREIGN KEY (reader_id) REFERENCES public.readers(reader_id);
 V   ALTER TABLE ONLY public.borrowed_books DROP CONSTRAINT borrowed_books_reader_id_fkey;
       public          postgres    false    3543    226    228            �           2606    17425 !   branches branches_manager_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.branches
    ADD CONSTRAINT branches_manager_id_fkey FOREIGN KEY (manager_id) REFERENCES public.employees(employee_id);
 K   ALTER TABLE ONLY public.branches DROP CONSTRAINT branches_manager_id_fkey;
       public          postgres    false    3535    216    218            �           2606    17420 "   employees employees_branch_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.employees
    ADD CONSTRAINT employees_branch_id_fkey FOREIGN KEY (branch_id) REFERENCES public.branches(branch_id);
 L   ALTER TABLE ONLY public.employees DROP CONSTRAINT employees_branch_id_fkey;
       public          postgres    false    3533    218    216            �           2606    17514    fines fines_borrow_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.fines
    ADD CONSTRAINT fines_borrow_id_fkey FOREIGN KEY (borrow_id) REFERENCES public.borrowed_books(borrow_id);
 D   ALTER TABLE ONLY public.fines DROP CONSTRAINT fines_borrow_id_fkey;
       public          postgres    false    228    3545    230            �           2606    17529 8   reader_notifications reader_notifications_reader_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.reader_notifications
    ADD CONSTRAINT reader_notifications_reader_id_fkey FOREIGN KEY (reader_id) REFERENCES public.readers(reader_id);
 b   ALTER TABLE ONLY public.reader_notifications DROP CONSTRAINT reader_notifications_reader_id_fkey;
       public          postgres    false    232    226    3543            �           2606    17478 )   readers readers_registered_branch_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.readers
    ADD CONSTRAINT readers_registered_branch_id_fkey FOREIGN KEY (registered_branch_id) REFERENCES public.branches(branch_id);
 S   ALTER TABLE ONLY public.readers DROP CONSTRAINT readers_registered_branch_id_fkey;
       public          postgres    false    216    3533    226            �   �   x�-OK
�0]gN���.��.�Pt!�"�-m����͍|��&���26�¡�xb@��n9[����4�8s�.�j�9S�ݡ��[5C��kGf)3����u�zh����e>{�7͂rapA4�E~BWG����Nq���h-N|V�}.�Щ��>����k�c �:e�W��"ٌD�4y�E      �   t  x���KnG��ݧ�	�c^˼�;d�� ��E�؎�s�"aڎ�+���Uӽ5�_��D�?��o�{أ���r.��}��*���_9�7.ૼ.�ʡܮOʱ|\[㧷�|U~wa�et?����6D� �,��!Fk�O�,���YD�XÆ��<�A���D�0�+ä�PfG�fI�0[
�YS0̞f���M9�	�������T����w�m��x�����6B}�8c�C��Sz��8��=�8����_���r}�C���:Mɧ�G��&�;M�w<��O}���w�=�������q�و����m�O��֥���.,�侮3F:b�%r�kw�#�8hs�>��A�t���y־3 .�w&�Alx��p���Ql8�bóĆ�(6D�� ��~��/��]��_�7��;N��eL�G��� S�7�� @��� U�c�j�(��1�j�je�,�
�E��%��|���EɏY��q���}���'>bW�W^�O��J�U�O���_��*�Tf?��
(UW��*�T]���}�{Z�ƿ]��>��sp\�<�L����:���������Z�s2b��9g#�v�<"�~3	"fD�"�8�Eaq��b`�����(,�~QX��b`��8�p��`e���l[}�yRNn���x�ULt��&���-�� ���-��Ak��"��2Y�%d�K�b�`6���fc	1K��XB��b6}��b6������$n��������q�xX���On��y��o5��Ӛ6F#��mLFN�ۘ��V�q䌆�� gr&A��̂��Y9��>����(|&G�39
��Q���I
��Q�L��gr>��=�x}Ё�i7_\?���st?�N�n�Ƕ懶ۺ�{�������:�����Bop�uWV;�̊c�Yq�3+��z���zű{��ؽ^q�^�8v�W8~��^q�^�8v�W��+^�	�^�s�����5xUNx�m�+g�\l�� �%��ڋۑFR�2�FI�ٸi$���&���Z������	��$N�$I��Y�8I�$q�&��h�I�8I�I�8N��q�4���o��$i'I�8Iǳ�4�Jp��o������8��      �   s   x�u���0D�C/�u�&���_G&�C"Yۏ�D�Wϟ9�L}���)���
{u�f� �
�I���*� ������Ў������)�S-��n~����:�����%M5	      ~   �   x���A
�@���)� %dv�N�aJ����-
�M��2M���o��͠E���3�{�W�(%�)"��y���92�'T2C����kq�PV�2��z�kf2v<��1R�X��"AC�d�,�Z����}$Ԝ�|D���č��j��[5#�|��P����MSi����}SۓJ��v�z ������{�v,cʒ�I��gH�Λ�'��8oG	�      �   �  x��V[R�@�^�BPJ�z_"'�0`�TRE%�$Ex�$߉1��1v���Q�gW�l��� ٥�ٞ�ie�����4l.���=����C}X�?ؑ+�8�u�3��Vi��a�e+|b��V��(�8S���}��z��ھ��u�����=!C{�wv��3�0�O��1�t�I�cr9$�6�%Le�D�o��ȱ� i�0���0:��H�� z�� y�&�:H�����������$>;P�(�"�ݜB�EU$E��)��@�`�� ���\C~�`P���ծ��	L��KB`W:��]+�5�ܗ�>��ɳ6���w�btUO}�uP(�w �)� �5�[���#)��q�u�ĭ��*Rk^T%tP*{�]7	�Y}���*8:#�mcu��C[#�\E�aݣ��b�����)��j-��	`:a���"�L����.i>#J��k�»�����0�������lq_�C��(t)m�
��@`/��>>���ɨ̃�[��|���غ'�Y��u	���\��AT�򚉄��T�8�/��H�d���N�&�Ȉ��\�[T������t����<1q󅊯o$؂�~��',��v'�QI�Ɏ9��7@��Z[2�V��D�9&]�Z���@X���M���4�	b�C������ͅ]&:�O�S��� UU�X��_�<��q>� ��fCrU����PUn��Dc.�{��֒n�tڻ<�s�6K_�q�(�t%n"�4�L;%p_�����w<��t~Q��Y���jn����]��:��f@e��Δ�@�M)��@��3拃���p�A�Nv�dX�n���ltc{mC���3Y�c�jH�ܻ��7�22fq��Y��6���p�n�ncg~�ϧe��Y\�-�M@��.����h�~��7G��X	,P�����ϲ��	���绻���y�M�S�$      �      x������ � �      �   >   x�3估�¾{.l������pa���ƋMv\l���b+�1��{.l��xaW� Nj!�      �      x������ � �      �   !  x�}U[n�0�&O�"��Kc;m�S��4H��z 7�%�u�F�}В��	��8���!�	�BW�6��a^���C<5m���(�ҔU�dm�;_g��->��O޷]�{,�a ��PڌQJ�mi���Ç�q-��%�:ZP^�V׍^�1^e
[aC\�>}2��ԢT���@i�� '�Heۙ����#�lؕ����'�[�E�rA�*�F!�Z��iW�~ЮE�� L�a��jM�
��]��W8x��L��P�.7�v�-�L�̤7,���B*�ˬP.
�!��Cκ܄K6hK�h.�I���}|�h=_j;��ސ7�P�#��X��k$�:W�K"X �3dS�`B�7�'�|J�Ӷa��Օ�s���耗�����aG#=�)��ҢL�D�8*�u�4<�T�#��]��3��<������� ��J��
��B8!7GebUqT�Z���wj�֎|Ŧ4H}��¨�ܱ4�.a��i��ҷ��I�rA��4�DD�^a�?�\ފW�����ߝ�\���{NR�y��c���O��m��~��Z�N�Ipt:�s���*�v��ᯱQS�(/ǳ�ku6K�tkל�=��h�facJ�L[r"�ɽPLg'��+��K9�y��f�T�VU�6O���[D@��xW��H������:e�Өy�{�����	zǑS��2�I�ĸ���WQ:G��Eܬ1	�xI�&��R}��^���oz#����8���lAU�m�>?���ǆ�L}��$����e�!��$W�e^&�R�ɻ&��/_Xk����;     