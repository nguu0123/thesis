#!/bin/bash/bin/bash
docker run --name postgres -e POSTGRES_USER=nguu0123 -e POSTGRES_PASSWORD=nguu0123456 -p 5432:5432 -v ${PWD}/postgres-docker:/var/lib/postgresql/data -v ${PWD}/init.sql:/docker-entrypoint-initdb.d/init.sql -d postgres
