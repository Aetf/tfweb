[logging]
level = "warn"

[defaults]
max_connections = 2000
client_idle_timeout = "10m"
backend_idle_timeout = "10m"
backend_connection_timeout = "2s"

[servers]

[servers.tfweb]
bind = "0.0.0.0:${port}"
protocol = "tcp"
balance = "leastconn"

[servers.tfweb.discovery]
interval = "2s"
kind = "exec"
exec_command = ["${tfweb_available}", "${piddir}"]

[servers.tfweb.healthcheck]
kind = "ping"
interval = "2s"
ping_timeout_duration = "500ms"
