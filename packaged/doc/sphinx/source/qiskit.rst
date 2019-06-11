Qiskit interoperations:
=================================


For more examples refer to :doc:`Tutorials/Qiskit`

Converters
------------------------
.. automodule:: qat.interop.qiskit.converters
    :members: to_qlm_circ, to_qiskit_circ
    

    


Providers
-------------------

QiskitQPU


.. autoclass:: qat.interop.qiskit.providers.QiskitQPU
    :members: submit_job, submit, set_backend
    

AsyncQiskitQPU



.. autoclass:: qat.interop.qiskit.providers.AsyncQiskitQPU
    :members: submit_job, set_backend, submit
    

QLMBackend



.. autoclass:: qat.interop.qiskit.providers.QLMBackend
    :members: set_qpu



Algorithms
------------------------
.. automodule:: qat.interop.qiskit.algorithms
    :members: shor_circuit, grover_circuit, qaoa_circuit


