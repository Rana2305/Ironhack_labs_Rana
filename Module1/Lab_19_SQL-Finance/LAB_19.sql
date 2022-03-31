select * from order_items;
select order_id, price from order_items
order by price desc;
use olist;
select order_id, price from order_items
order by price asc;
select min(shipping_limit_date), max(shipping_limit_date) from order_items;
select customer_state, count(customer_unique_id) from customers
group by customer_state
order by count(customer_unique_id) desc;
select customer_state, count(customer_id) from customers
group by customer_state
limit 1;
select customer_city, count(customer_id) from customers
where customer_state = "SP"
order by count(customer_id);
select * from closed_deals;
select count(distinct(business_segment)) from closed_deals;
select sum(declared_monthly_revenue), business_segment from closed_deals
group by business_segment
having count(business_segment) >1
order by sum(declared_monthly_revenue) desc
limit 3;
select * from order_reviews;
select sum(distinct(review_score)) from order_reviews;

select * from olist.order_reviews;
alter table olist.order_reviews ADD column category_review char(50);
SELECT review_score, count(review_score),
   CASE review_score
      WHEN 1 THEN 'Bad'
      WHEN 2 THEN 'Poor'
      WHEN 3 THEN 'Satisfactory'
	  WHEN 4 THEN 'Good'
      WHEN 5 THEN 'Perfect'
   END AS review_categ
 FROM olist.order_reviews group by review_categ order by count(review_score) desc;

select review_score, count(review_score) from order_reviews
group by review_score
order by count(review_score) desc
limit 1;

select * from customers;


