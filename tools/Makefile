PYTHON:= python3
CUDA_VERSION:= 12.1
PYTORCH_VERSION:= 2.5.1
DOT:= .
.PHONY: all clean show_variables s3prl_vc pytorch uninstall_pytorch monotonic_align

all: show_variables virtualenv.done pytorch.done jatts.done

jatts: jatts.done

s3prl_vc: s3prl_vc.done

pytorch: pytorch.done

show_variables:
	@echo PYTHON=$(PYTHON)
	@echo CUDA_VERSION=$(CUDA_VERSION)
	@echo PYTORCH_VERSION=$(PYTORCH_VERSION)

virtualenv.done: show_variables
	test -d venv || $(PYTHON) -m venv venv
	. venv/bin/activate; cd ../; pip install -U pip
	# install numpy here since python3.6 is not supported in > 1.20
	. venv/bin/activate; cd ../; pip install numpy
	touch virtualenv.done

pytorch.done: virtualenv.done
ifeq ($(CUDA_VERSION),)
	. venv/bin/activate; pip install torch==$(PYTORCH_VERSION) \
		-f https://download.pytorch.org/whl/cpu/stable.html
else
	. venv/bin/activate; pip install torch==$(PYTORCH_VERSION) \
		-f https://download.pytorch.org/whl/cu$(subst $(DOT),,$(CUDA_VERSION))/torch_stable.html
endif
	touch pytorch.done

jatts.done: virtualenv.done pytorch.done
	. venv/bin/activate; cd ../; pip install -e .
	. venv/bin/activate; cd ../; pip install git+https://github.com/speechbrain/speechbrain.git@develop
	. venv/bin/activate; cd ../; pip install git+https://github.com/rinnakk/nue-asr.git
	touch jatts.done

apex.done: virtualenv.done pytorch.done
	git clone https://github.com/NVIDIA/apex.git
	. venv/bin/activate; cd apex; \
		pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
	touch apex.done

monotonic_align:
	. venv/bin/activate; cd ../jatts/monotonic_align; python setup.py build_ext --inplace; cd ../

clean:
	rm -fr venv apex *.done
	find -iname "*.pyc" -delete

uninstall_pytorch:
	. venv/bin/activate; cd ../; pip uninstall torch
