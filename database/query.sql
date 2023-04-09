select * from predict
inner join used on predict.id = used.activityid
inner join data on used.entityid = data.id

