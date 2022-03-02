SELECT * FROM apple;
select prime_genre, count(track_name)   from apple
group by prime_genre;
select prime_genre, avg(user_rating) as bestrating from apple
group by prime_genre
order by bestrating desc;
select prime_genre, sum(track_name) as mostapps from apple
group by prime_genre
order by mostapps desc;
select track_name, rating_count_tot, prime_genre from apple
order by rating_count_tot desc
limit 10;
select track_name, user_rating,
select track_name, user_rating, prime_genre from apple
order by user_rating desc
limit 10;
select track_name, concat(user_rating, '' , cont_rating)  as Newcol from apple
order by Newcol desc
limit 3;
select track_name from apple
where user_rating = 5
order by rating_count_tot desc
limit 3;
select track_name, user_rating, rating_count_tot, price from apple
order by user_rating desc;
select track_name, user_rating, cont_rating from apple
order by user_rating desc, cont_rating desc
limit 3;
use appledatabase;
select track_name, user_rating, cont_rating, price from apple
order by user_rating desc;
SELECT track_name,  user_rating, rating_count_tot, price FROM apple
ORDER by user_rating desc , price desc;
SELECT prime_genre,  user_rating, rating_count_tot, price FROM apple 
group by prime_genre
ORDER by user_rating desc;