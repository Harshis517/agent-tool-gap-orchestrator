#!/usr/bin/env bash
set -euo pipefail

# --- sanity ---
echo "Java:"; java -version || true

# dirs for noexec /tmp environments
mkdir -p "$HOME/.cache/jansi" "$HOME/.cache/java" "$HOME/.m2_isolated"
export MAVEN_OPTS="-Djansi.tmpdir=$HOME/.cache/jansi -Djava.io.tmpdir=$HOME/.cache/java -Dorg.fusesource.jansi.disable=true -Djansi.mode=off"

# make sure we have the wrapper (locks Maven version for consistency)
if [ ! -f ./mvnw ]; then
  mvn -N io.takari:maven:wrapper -Dmaven=3.9.6
fi
chmod +x mvnw

# ensure private jar is installed into the SAME isolated repo
JAR_DEFAULT="$HOME/archemy-security-1.0-SNAPSHOT.jar"
if [ -f "$JAR_DEFAULT" ]; then
  echo "[INFO] Installing private jar $JAR_DEFAULT into isolated repo"
  mvn -q org.apache.maven.plugins:maven-install-plugin:3.1.1:install-file \
    -Dmaven.repo.local="$HOME/.m2_isolated" \
    -Dfile="$JAR_DEFAULT" \
    -DgroupId=com.archemy \
    -DartifactId=archemy-security \
    -Dversion=1.0-SNAPSHOT \
    -Dpackaging=jar
else
  echo "[WARN] Private jar not found at $JAR_DEFAULT"
  echo "       If you have it elsewhere, install with:"
  echo "       mvn -q org.apache.maven.plugins:maven-install-plugin:3.1.1:install-file \\"
  echo "         -Dmaven.repo.local=\$HOME/.m2_isolated -Dfile=/path/to/archemy-security-1.0-SNAPSHOT.jar \\"
  echo "         -DgroupId=com.archemy -DartifactId=archemy-security -Dversion=1.0-SNAPSHOT -Dpackaging=jar"
fi

# try spring-boot:run with explicit plugin version (avoids bad cache)
./mvnw -U -q -Dmaven.repo.local="$HOME/.m2_isolated" -DskipTests \
  org.springframework.boot:spring-boot-maven-plugin:3.3.3:run \
  -Dspring-boot.run.jvmArguments="-Xms64m -Xmx256m -XX:MaxMetaspaceSize=128m -Dserver.port=\${SERVER_PORT:-9090} -Dlicensing.valid.keys=\${LIC_KEYS:-ABC-123} -Djava.io.tmpdir=$HOME/.cache/java" \
|| {
  echo "[WARN] spring-boot:run failed; falling back to building and running the jar…"
  ./mvnw -U -q -Dmaven.repo.local="$HOME/.m2_isolated" -DskipTests package
  exec java -Xms64m -Xmx256m -XX:MaxMetaspaceSize=128m \
    -Dserver.port="\${SERVER_PORT:-9090}" \
    -Dlicensing.valid.keys="\${LIC_KEYS:-ABC-123}" \
    -Djava.io.tmpdir="$HOME/.cache/java" \
    -jar target/licensing-service-*.jar
}
