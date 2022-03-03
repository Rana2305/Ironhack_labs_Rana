use publications;
select * from titleauthor;
select * from authors;
select authors.au_lname, authors.au_fname as author_name, COUNT(titleauthor.title_id) AS Number_of_books
From authors
LEFT JOIN titleauthor
ON authors.au_id = titleauthor.au_id
GROUP BY author_name;
select * from titles;
CREATE temporary Table summary
SELECT authors.au_id, authors.au_lname, authors.au_fname, 
titleauthor.title_id, titles.title, titles.pub_id, publishers.pub_name
FROM authors
left join titleauthor
on titleauthor.au_id=authors.au_id
left join titles
on titles.title_id=titleauthor.title_id
left join publishers
on titles.pub_id=publishers.pub_id
where titles.title_id is not null;
select * from summary;
select au_id, au_lname as Last_name, au_fname as first_name, count(title), pub_name from summary
group by au_id
order by count(title) desc;
select * from sales;
select authors.au_id, authors.au_lname as last_name, authors.au_fname as first_name,
sales.qty as total
from authors
left join titleauthor
on  titleauthor.au_id=authors.au_id
left join sales
on sales.title_id = titleauthor.title_id
group by last_name
order by total desc
limit 3;

select * from authors;
select authors.au_id, authors.au_lname as last_name, coalesce(qty,0) as total,  authors.au_fname as first_name
from authors
left join titleauthor
on  titleauthor.au_id=authors.au_id
left join sales
on sales.title_id = titleauthor.title_id
group by last_name
order by total desc;







