"""
Employee table
get the cumulative sum of an employee's salary over a period of 3 months but exclude the most recent month.

The result should be displayed by 'Id' ascending, and then by 'Month' descending.

Input

| Id | Month | Salary |
|----|-------|--------|
| 1  | 1     | 20     |
| 2  | 1     | 20     |
| 1  | 2     | 30     |
| 2  | 2     | 30     |
| 3  | 2     | 40     |
| 1  | 3     | 40     |
| 3  | 3     | 60     |
| 1  | 4     | 60     |
| 3  | 4     | 70     |
Output

| Id | Month | Salary |
|----|-------|--------|
| 1  | 3     | 90     |
| 1  | 2     | 50     |
| 1  | 1     | 20     |
| 2  | 1     | 20     |
| 3  | 3     | 100    |
| 3  | 2     | 40     |
"""

SELECT E1.Id, E1.month, (IFNULL(E1.Salary, 0) + IFNULL(E2.Salary,0)+ IFNULL(E3.Salary,0))AS Salary
FROM 
(select Id, max(month) AS month FROM Employee 
GROUP BY Id HAVING count(month) > 1) AS maxmonth
LEFT JOIN
Employee E1 ON (E1.Id = maxmonth.Id AND maxmonth.month > E1.month)
LEFT JOIN
Employee E2 ON (E2.Id = E1.Id AND E2.month = E1.month -1)
LEFT JOIN
Employee E3 ON (E3.Id = E1.Id AND E3.month = E1.month - 2)
ORDER BY id asc, month desc


