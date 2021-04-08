pipeline {
	agent {
		docker {
			alwaysPull true
			image 'gitlab2.informatik.uni-wuerzburg.de:4567/dmir/dallmann/recommender/asme-dev:latest'
			registryCredentialsId 'gitlab-asme'
			registryUrl 'https://gitlab2.informatik.uni-wuerzburg.de:4567/dmir/dallmann/recommender/asme-dev:latest'
			reuseNode true
			args '--entrypoint=\'\''
		}
	}
	triggers {
		pollSCM 'H/10 * * * *'
	}
	options {
		disableConcurrentBuilds()
		buildDiscarder(logRotator(numToKeepStr: '15', artifactNumToKeepStr: '15'))
		timeout(time: 2, unit: 'HOURS')
	}
	stages {
		stage('Poetry Install') {
			steps {
				sh 'export HOME=`pwd`/env; poetry install --no-root'
			}
		}
		stage('Run Tests') {
			steps {
				sh 'export HOME=`pwd`/env; export PYTHONPATH=`pwd`/src; poetry run pytest -v --junitxml=reports/result.xml'
			}
			post {
				always {
					junit 'reports/result.xml'
				}
			}
		}
	}
}
