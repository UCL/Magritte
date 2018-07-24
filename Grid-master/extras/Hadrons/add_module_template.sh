#!/usr/bin/env bash

if (( $# != 1 && $# != 2)); then
    echo "usage: `basename $0` <module name> [<namespace>]" 1>&2
    exit 1
fi
NAME=$1
NS=$2

if (( $# == 1 )); then
	if [ -e "Modules/${NAME}.cc" ] || [ -e "Modules/${NAME}.hpp" ]; then
	    echo "error: files Modules/${NAME}.* already exists" 1>&2
	    exit 1
	fi
	sed "s/___FILEBASENAME___/${NAME}/g" Modules/templates/Module_tmp.hpp.template > Modules/${NAME}.hpp
elif (( $# == 2 )); then
	mkdir -p Modules/${NS}
	if [ -e "Modules/${NS}/${NAME}.cc" ] || [ -e "Modules/${NS}/${NAME}.hpp" ]; then
	    echo "error: files Modules/${NS}/${NAME}.* already exists" 1>&2
	    exit 1
	fi
	TMPCC=".${NS}.${NAME}.tmp.cc"
	TMPHPP=".${NS}.${NAME}.tmp.hpp"
	sed "s/___FILEBASENAME___/${NAME}/g" Modules/templates/Module_tmp_in_NS.hpp.template > ${TMPHPP}
	sed "s/___NAMESPACE___/${NS}/g" ${TMPHPP} > Modules/${NS}/${NAME}.hpp
	rm -f ${TMPCC} ${TMPHPP}
fi
./make_module_list.sh
