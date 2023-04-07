#!/bin/bash
docker run --name pgadmin-nguu0123 -p 5051:80 -e "PGADMIN_DEFAULT_EMAIL=nguu0123@gmail.com" -e "PGADMIN_DEFAULT_PASSWORD=nguu0123456" -d dpage/pgadmin4
