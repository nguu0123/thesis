#!/bin/bash

  /bin/sh -ec 'serve run deployment:ensemble' &
  /bin/sh -ec 'sleep 10 && cd client && python3 client.py'