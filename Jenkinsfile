#!/usr/local/bin/groovy

// ---------------------------------------------------------------------------
//
// GROOVY GLOBALS
//
// ---------------------------------------------------------------------------
def QLM_VERSION_FOR_DOCKER_IMAGE = "1.1.0"

//def REFERENCE_DOCKER = "yes"
def REFERENCE_DOCKER = "no"

def OS
def LABEL
def OSLABEL
def JOB_TYPE
def DOCKER_IMAGE

LABEL = "master"

try {
    if ("$UI_OSVERSION".startsWith("7"))
        OS = "el7"
    else if ("$UI_OSVERSION".startsWith("8"))
        OS = "el8"
} catch (e) {
    echo "***** UI_OSVERSION undefined; setting it to 8.2 *****"
    UI_OSVERSION = 8.2
    OS = "el8"
}

OSLABEL  = "rhel$UI_OSVERSION"
if (!env.BRANCH_NAME) {
    env.BRANCH_NAME = "master"
}

if ("$REFERENCE_DOCKER".contains("yes"))
    DOCKER_IMAGE = "qlm-reference-${QLM_VERSION_FOR_DOCKER_IMAGE}-${OSLABEL}:latest"
else
    DOCKER_IMAGE = "qlm-devel-${QLM_VERSION_FOR_DOCKER_IMAGE}-${OSLABEL}:latest"

HOST_NAME = InetAddress.getLocalHost().getHostName()

// Expose OS to bash and groovy functions
env.OS = "$OS"
env.HOST_NAME = "$HOST_NAME"
env.NIGHTLY_BUILD = params.NIGHTLY_BUILD

// Show the parameters
echo "\
JOB_NAME     = ${JOB_NAME}\n\
BRANCH_NAME  = ${BRANCH_NAME}\n\
JOB_BASE_NAME= ${JOB_BASE_NAME}\n\
UI_OSVERSION = ${UI_OSVERSION}\n\
OS           = ${OS}\n\
OSLABEL      = ${OSLABEL}\n\
DOCKER_IMAGE = ${DOCKER_IMAGE}\n\
HOST_NAME    = ${HOST_NAME}"


// ---------------------------------------------------------------------------
//
// Configure some of the job properties
//
// ---------------------------------------------------------------------------
properties([
    [$class: 'JiraProjectProperty'],
    [$class: 'EnvInjectJobProperty',
        info: [
            loadFilesFromMaster: false,
            propertiesContent: '''
                someList=
            ''',
            secureGroovyScript: [
                classpath: [],
                sandbox: false,
                script: ''
            ]
        ],
        keepBuildVariables: true,
        keepJenkinsSystemVariables: true,
        on: true
    ],
    buildDiscarder(logRotator(artifactDaysToKeepStr: '', artifactNumToKeepStr: '', daysToKeepStr: '', numToKeepStr: '25')),
    disableConcurrentBuilds(),
    pipelineTriggers([pollSCM('')]),
    parameters([
        [$class: 'ChoiceParameter', choiceType: 'PT_SINGLE_SELECT', description: '', filterLength: 1, filterable: false, name: 'UI_VERSION', randomName: 'choice-parameter-266216487624195',
            script: [
                $class: 'ScriptlerScript',
                parameters: [
                    [$class: 'org.biouno.unochoice.model.ScriptlerScriptParameter', name: 'job_name',    value: "${JOB_NAME}"],
                    [$class: 'org.biouno.unochoice.model.ScriptlerScriptParameter', name: 'host_name',   value: "${HOST_NAME}"],
                    [$class: 'org.biouno.unochoice.model.ScriptlerScriptParameter', name: 'branch_name', value: "${BRANCH_NAME}"]
                ],
                scriptlerScriptId: 'ReturnNextVersions.groovy'
            ]
        ],
        [$class: 'ChoiceParameter', choiceType: 'PT_RADIO', description: '', filterLength: 1, filterable: false, name: 'UI_OSVERSION', randomName: 'choice-parameter-744322351209535',
            script: [
                $class: 'GroovyScript',
                fallbackScript: [
                    classpath: [],
                    sandbox: false,
                    script: ' '
                ],
                script: [
                    classpath: [],
                    sandbox: false,
                    script: '''
                        return ['7.8', '8.2:selected']
                    '''
                ]
            ]
        ],
        [$class: 'ChoiceParameter', choiceType: 'PT_RADIO', description: '', filterLength: 1, filterable: false, name: 'UI_PRODUCT', randomName: 'choice-parameter-2765756439171963',
            script: [
                $class: 'ScriptlerScript',
                parameters: [
                    [$class: 'org.biouno.unochoice.model.ScriptlerScriptParameter', name: 'job_name', value: "${JOB_NAME}"]
                ],
                scriptlerScriptId: 'selectProductsToBuild.groovy'
            ]
        ],
        [$class: 'CascadeChoiceParameter', choiceType: 'PT_RADIO', description: '', filterLength: 1, filterable: false, name: 'UI_TESTS', randomName: 'choice-parameter-851409291728428',
            referencedParameters: 'UI_OSVERSION,BRANCH_NAME',
            script: [
                $class: 'GroovyScript',
                fallbackScript: [
                    classpath: [],
                    sandbox: false,
                    script: ' '
                ],
                script: [
                    classpath: [],
                    sandbox: false,
                    script: '''
                        return ['Run tests:selected', 'Skip tests']
                    '''
                ]
            ]
        ]

    ])
])


// ---------------------------------------------------------------------------
//
// Declarative pipeline
//
// ---------------------------------------------------------------------------
pipeline
{
    agent {
        docker {
            label "${LABEL}"
            image "qlm-devel-${QLM_VERSION_FOR_DOCKER_IMAGE}-rhel8.2:latest"
            args '-v /data/jenkins/.ssh:/data/jenkins/.ssh -v /opt/qlmtools:/opt/qlmtools -v /var/www/repos:/var/www/repos'
            alwaysPull false
        }
    }

    options {
        ansiColor('xterm')
    }

    environment {
        RUN_BY_JENKINS=1

        BASEDIR           = "$WORKSPACE"
        QATDIR            = "$BASEDIR/qat"
        QAT_REPO_BASEDIR  = "$BASEDIR"
        INSTALL_DIR       = "$BASEDIR/install"
        RUNTIME_DIR       = "$BASEDIR/runtime"

        BLACK   = '\033[30m' ; B_BLACK   = '\033[40m'
        RED     = '\033[31m' ; B_RED     = '\033[41m'
        GREEN   = '\033[32m' ; B_GREEN   = '\033[42m'
        YELLOW  = '\033[33m' ; B_YELLOW  = '\033[43m'
        BLUE    = '\033[34m' ; B_BLUE    = '\033[44m'
        MAGENTA = '\033[35m' ; B_MAGENTA = '\033[45m'
        CYAN    = '\033[36m' ; B_CYAN    = '\033[46m'
        WHITE   = '\033[97m' ; B_WHITE   = '\033[107m'
        BOLD    = '\033[1m'  ; UNDERLINE = '\033[4m'
        RESET   = '\033[0m'

        BUILD_CAUSE      =  currentBuild.getBuildCauses()[0].shortDescription.toString()
        BUILD_CAUSE_NAME =  currentBuild.getBuildCauses()[0].userName.toString()

        REPO_NAME = sh returnStdout: true, script: '''set +x
            JOB_NAME=${JOB_NAME%%/*}
            if [[ $JOB_NAME =~ ^qat-.*-.*$ ]]; then
                echo -n ${JOB_NAME%-*}
            else
                echo -n $JOB_NAME
            fi
        '''

        JOB_QUALIFIER = sh returnStdout: true, script: '''set +x
            JOB_QUALIFIER=${JOB_NAME#*qat-}
            JOB_QUALIFIER=${JOB_QUALIFIER#*-}
            n=${JOB_NAME//[^-]}
            if ((${#n} > 1)); then
                echo -n "/${JOB_QUALIFIER%%/*}"
            else
                echo -n "/"
            fi
        '''

        QUALIFIED_REPO_NAME = sh returnStdout: true, script: '''set +x
            x=$JOB_NAME
            if [[ $JOB_NAME =~ ^qat-.*-.*$ ]]; then
                x=${JOB_NAME%-*}
                if [[ $x =~ ^qat-.*-.*$ ]]; then
                    x=${x%-*}
                fi
            fi
            REPO_NAME=${x%%/*}
            JOB_QUALIFIER=${JOB_NAME#*qat-}
            JOB_QUALIFIER=${JOB_QUALIFIER#*-}
            n=${JOB_NAME//[^-]}
            if ((${#n} > 1)); then
                JOB_QUALIFIER="${JOB_QUALIFIER%%/*}"
                echo -n "${REPO_NAME}-${JOB_QUALIFIER}"
            else
                echo -n "${REPO_NAME}"
            fi
        '''

        CURRENT_OS        = "el8"
        CURRENT_PLATFORM  = "linux"
        DEPENDENCIES_OS   = "$OS"
    } 


    stages
    {
        stage('Init')
        {
            steps {
                echo "${MAGENTA}${BOLD}[INIT]${RESET}"
                echo "\
BASEDIR             = ${BASEDIR}\n\
QATDIR              = ${QATDIR}\n\
QAT_REPO_BASEDIR    = ${QAT_REPO_BASEDIR}\n\
\n\
BUILD_CAUSE         = ${BUILD_CAUSE}\n\
BUILD_CAUSE_NAME    = ${BUILD_CAUSE_NAME}\n\
\n\
REPO_NAME           = ${REPO_NAME}\n\
\n\
JOB_QUALIFIER       = ${JOB_QUALIFIER}\n\
QUALIFIED_REPO_NAME = ${QUALIFIED_REPO_NAME}\n\
NIGHTLY_BUILD       = ${NIGHTLY_BUILD}\n\
"

                sh '''set +x
                    mkdir -p $REPO_NAME
                    mv * $REPO_NAME/ 2>/dev/null  || true
                    mv .* $REPO_NAME/ 2>/dev/null || true

                    ATOS_GIT_BASE_URL=ssh://bitbucketbdsfr.fsc.atos-services.net:7999/brq
                    if [[ $HOST_NAME =~ qlmci2 ]]; then
                        ATOS_GIT_BASE_URL=ssh://qlmjenkins@qlmgit.usrnd.lan:29418/qlm
                    fi

                    # Clone qat repo
                    echo -e "--> Cloning qat, branch=$BRANCH_NAME  [$ATOS_GIT_BASE_URL] ..."
                    cmd="git clone --single-branch --branch $BRANCH_NAME $ATOS_GIT_BASE_URL/qat"
                    echo "> $cmd"
                    eval $cmd

                    # Install wheels dependencies
                    if [[ $REPO_NAME = myqlm-interop ]]; then
                        sudo python3 -m pip install --upgrade pip
                        sudo pip3 install -r $WORKSPACE/qat/share/misc/myqlm-interop-requirements.txt || true
                    fi
                '''
                script {
                    print "Loading groovy functions ..."
                    support_methods         = load "${QATDIR}/jenkins_methods/support"
                    build_methods           = load "${QATDIR}/jenkins_methods/build"
                    install_methods         = load "${QATDIR}/jenkins_methods/install"
                    static_analysis_methods = load "${QATDIR}/jenkins_methods/static_analysis"
                    test_methods            = load "${QATDIR}/jenkins_methods/tests"
                    packaging_methods       = load "${QATDIR}/jenkins_methods/packaging"

                    // Set a few badges for the build
                    support_methods.badges()
                }
            }
        } // Init


        stage('Versioning')
        {
            steps {
                echo "${MAGENTA}${BOLD}[VERSIONING]${RESET}"
                script {
                    VERSION = sh returnStdout: true, script: '''set +x
                        if [[ -n $UI_VERSION ]]; then
                            VERSION="$UI_VERSION"
                        else
                            # UI_VERSION can be null from curl command
                            if [[ -r qat/share/versions/$REPO_NAME.version ]]; then
                                VERSION="$(cat qat/share/versions/$REPO_NAME.version)"
                            else
                                echo -e "\n**** No qat/share/versions/$REPO_NAME.version file"
                                exit 1
                            fi
                        fi
                        if [[ $BRANCH_NAME != rc ]]; then
                            VERSION=${VERSION}.${BUILD_NUMBER}
                        fi
                        echo -n $VERSION
                    '''
                    env.VERSION="$VERSION"
                    echo "(wheel) -> ${VERSION}"

                    sh '''set +x
                        sed -i "s/version=.*/version=\\"$VERSION\\",/" $WORKSPACE/$REPO_NAME/setup.py
                    '''

                    // Commit the new version
                    sh '''set +x
                        # Commit a change in versioning if any
                        # Note: Use of qlmjenkins will not trigger a new build on commit
                        cd $WORKSPACE/qat/share/versions
                        committer_name=$(git log --format="%an" | head -1)
                        git config --local user.name  "qlmjenkins"
                        git config --local user.email "atos@noreply.com"
                        if [[ -n $UI_VERSION && $(cat $REPO_NAME.version 2>/dev/null) != $UI_VERSION ]]; then
                            echo -e "\n${CYAN}--> Committing version...${RESET}"
                            echo -n "$UI_VERSION" >$REPO_NAME.version
                            git remote -v
                            git add $REPO_NAME.version
                            if git commit -m "Version change [$REPO_NAME,$UI_VERSION,$committer_name]"; then
                                git pull origin $BRANCH_NAME
                                if git push origin HEAD:$BRANCH_NAME; then
                                    echo -e "\nThe new version [$UI_VERSION] has been pushed"
                                fi
                            fi
                        fi
                    '''
                }
                buildName "${VERSION}-${OS}"
            }
        } // Versioning


        stage('Install')
        {
            steps {
                script {
                    sh '''set +x
                        source /usr/local/bin/qatenv
                        mkdir -p $INSTALL_DIR/lib64/python3.6/site-packages/
                        cmd="cp -r ${REPO_NAME}/qat $INSTALL_DIR/lib64/python3.6/site-packages/"
                        echo -e "\n> $cmd"
                        $cmd

                        # Save the artifact(s)
                        echo -e "\nCreating ${REPO_NAME}-${VERSION}-${CURRENT_PLATFORM}.${CURRENT_OS}.tar.gz"
                        cd $INSTALL_DIR && tar cfz $WORKSPACE/${REPO_NAME}-${VERSION}-${CURRENT_PLATFORM}.${CURRENT_OS}.tar.gz .
                    '''
                    archiveArtifacts artifacts: "${REPO_NAME}-${VERSION}-${CURRENT_PLATFORM}.${CURRENT_OS}.tar.gz", onlyIfSuccessful: true
                }
            }
        } // Install


        stage('Tests-dependencies')
        {
            when {
                expression { if (env.UI_TESTS.toLowerCase().contains("skip")) { return false } else { return true } }
            }
            steps {
                script {
                    env.stage = "tests"
                    support_methods.restore_tarballs_dependencies(env.stage)
                }
            }
        } // Tests-dependencies


        stage('Tests')
        {
            when {
                expression { if (env.UI_TESTS.toLowerCase().contains("skip")) { return false } else { return true } }
            }
            environment {
                INSTALL_DIR                 = "$WORKSPACE/install"
                TESTS_REPORTS_DIR           = "$REPO_NAME/tests/reports"
                TESTS_REPORTS_DIR_JUNIT     = "$TESTS_REPORTS_DIR/junit"
                TESTS_REPORTS_DIR_GTEST     = "$TESTS_REPORTS_DIR/gtest"
                TESTS_REPORTS_DIR_CUNIT     = "$TESTS_REPORTS_DIR/cunit"
                GTEST_OUTPUT                = "xml:$WORKSPACE/$TESTS_REPORTS_DIR_GTEST/"
                TESTS_REPORTS_DIR_COVERAGE  = "$TESTS_REPORTS_DIR/coverage"
                TESTS_REPORTS_DIR_VALGRIND  = "$TESTS_REPORTS_DIR/valgrind"
                VALGRIND_ARGS               = "--fair-sched=no --child-silent-after-fork=yes --tool=memcheck --xml=yes --xml-file=$WORKSPACE/$TESTS_REPORTS_DIR_VALGRIND/report.xml --leak-check=full --show-leak-kinds=all --show-reachable=no --track-origins=yes --run-libc-freeres=no --gen-suppressions=all --suppressions=$QATDIR/share/misc/valgrind.supp"
            }
            steps {
                echo "${MAGENTA}${BOLD}[TESTS]${RESET}"
                script {
                    catchError(buildResult: 'UNSTABLE', stageResult: 'UNSTABLE') {
                        sh '''set +x
                            source /usr/local/bin/qatenv
                            cp qat/share/misc/pytest.ini $REPO_NAME/tests 2>/dev/null
                            mkdir -p $WORKSPACE/$TESTS_REPORTS_DIR_JUNIT
                            cd $REPO_NAME/tests
                            cmd="python3 -m pytest -v --junitxml=reports/junit/report.xml ."
                            echo -e "\n> $cmd"
                            $cmd
                        '''
                    }
                    test_methods.tests_reporting()
                }
            }
        } // Tests


        stage("WHEEL")
        {
            steps {
                echo "${MAGENTA}${BOLD}[WHEEL]${RESET}"
                script {
                    sh '''set +x
                        source /usr/local/bin/qatenv
                        cd $WORKSPACE/$REPO_NAME

                        echo -e "\n${CYAN}Building myQLM wheels...${RESET}"
                        cmd="$PYTHON setup.py bdist_wheel"
                        echo -e "\n> ${GREEN}$cmd${RESET}"
                        $cmd
 
                        echo -e "\n\n${MAGENTA}WHEEL packaged files list${RESET}"
                        find dist -type f -exec echo -e "$BLUE"{}"${RESET}" \\; -exec unzip -l {} \\;
    
                        echo -e "\n\n${MAGENTA}METADATA file${RESET}"
                        find dist -type f -exec unzip -p {} ${REPO_NAME//-/_}-$VERSION.dist-info/METADATA \\;
                    '''
                    // Save the source tarball and wheel artifacts
                    archiveArtifacts artifacts: "${REPO_NAME}/dist/*.whl", onlyIfSuccessful: true
                }
            }
        } // wheel
    } // stages


    post
    {
        always
        {
            echo "${MAGENTA}${BOLD}[POST]${RESET}"
            script {
                sh '''set +x
                    rm -f tarballs_artifacts/.*.artifact 2>/dev/null
                '''
                // Send emails only if not started by upstream (qat pipeline)
                if (!BUILD_CAUSE.contains("upstream")) {
                    emailext body:
                        "${BUILD_URL}",
                        recipientProviders: [[$class:'CulpritsRecipientProvider'],[$class:'RequesterRecipientProvider']],
                        subject: "${BUILD_TAG} - ${currentBuild.result}"
                }
            }
        } // always
    } // post
} // pipeline

