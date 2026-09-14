-- ============================================================
-- Enterprise AI Data Analyst — SQL Schema
-- Member 1: Data Engineering & SQL Lead
-- Source dataset: UCI Online Retail II
-- Engine: SQLite (portable, works directly inside Colab)
-- ============================================================

DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;

-- Dimension: customers
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    country     TEXT NOT NULL
);

-- Dimension: products
CREATE TABLE products (
    stock_code  TEXT PRIMARY KEY,
    description TEXT
);

-- Fact/header: orders (one row per invoice)
CREATE TABLE orders (
    invoice_no    TEXT PRIMARY KEY,
    customer_id   INTEGER,
    invoice_date  DATETIME NOT NULL,
    is_cancelled  INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Fact/detail: order line items
CREATE TABLE order_items (
    item_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_no  TEXT NOT NULL,
    stock_code  TEXT NOT NULL,
    quantity    INTEGER NOT NULL,
    unit_price  REAL NOT NULL,
    revenue     REAL NOT NULL,
    FOREIGN KEY (invoice_no) REFERENCES orders(invoice_no),
    FOREIGN KEY (stock_code) REFERENCES products(stock_code)
);

CREATE INDEX idx_orders_customer   ON orders(customer_id);
CREATE INDEX idx_orders_date       ON orders(invoice_date);
CREATE INDEX idx_items_invoice     ON order_items(invoice_no);
CREATE INDEX idx_items_stockcode   ON order_items(stock_code);
