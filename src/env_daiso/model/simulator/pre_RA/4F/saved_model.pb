�	
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
�
Adam/v/dense_4079/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/dense_4079/bias
}
*Adam/v/dense_4079/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_4079/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_4079/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/dense_4079/bias
}
*Adam/m/dense_4079/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_4079/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_4079/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_nameAdam/v/dense_4079/kernel
�
,Adam/v/dense_4079/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_4079/kernel*
_output_shapes

: *
dtype0
�
Adam/m/dense_4079/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_nameAdam/m/dense_4079/kernel
�
,Adam/m/dense_4079/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_4079/kernel*
_output_shapes

: *
dtype0
�
Adam/v/dense_4078/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/v/dense_4078/bias
}
*Adam/v/dense_4078/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_4078/bias*
_output_shapes
: *
dtype0
�
Adam/m/dense_4078/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/m/dense_4078/bias
}
*Adam/m/dense_4078/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_4078/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_4078/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *)
shared_nameAdam/v/dense_4078/kernel
�
,Adam/v/dense_4078/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_4078/kernel*
_output_shapes
:	� *
dtype0
�
Adam/m/dense_4078/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *)
shared_nameAdam/m/dense_4078/kernel
�
,Adam/m/dense_4078/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_4078/kernel*
_output_shapes
:	� *
dtype0
�
Adam/v/dense_4077/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam/v/dense_4077/bias
~
*Adam/v/dense_4077/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_4077/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_4077/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam/m/dense_4077/bias
~
*Adam/m/dense_4077/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_4077/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_4077/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*)
shared_nameAdam/v/dense_4077/kernel
�
,Adam/v/dense_4077/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_4077/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_4077/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*)
shared_nameAdam/m/dense_4077/kernel
�
,Adam/m/dense_4077/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_4077/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/dense_4076/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam/v/dense_4076/bias
~
*Adam/v/dense_4076/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_4076/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_4076/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam/m/dense_4076/bias
~
*Adam/m/dense_4076/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_4076/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_4076/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*)
shared_nameAdam/v/dense_4076/kernel
�
,Adam/v/dense_4076/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_4076/kernel*
_output_shapes
:	@�*
dtype0
�
Adam/m/dense_4076/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*)
shared_nameAdam/m/dense_4076/kernel
�
,Adam/m/dense_4076/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_4076/kernel*
_output_shapes
:	@�*
dtype0
�
Adam/v/dense_4075/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/v/dense_4075/bias
}
*Adam/v/dense_4075/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_4075/bias*
_output_shapes
:@*
dtype0
�
Adam/m/dense_4075/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/m/dense_4075/bias
}
*Adam/m/dense_4075/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_4075/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_4075/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*)
shared_nameAdam/v/dense_4075/kernel
�
,Adam/v/dense_4075/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_4075/kernel*
_output_shapes

:@*
dtype0
�
Adam/m/dense_4075/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*)
shared_nameAdam/m/dense_4075/kernel
�
,Adam/m/dense_4075/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_4075/kernel*
_output_shapes

:@*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
v
dense_4079/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_4079/bias
o
#dense_4079/bias/Read/ReadVariableOpReadVariableOpdense_4079/bias*
_output_shapes
:*
dtype0
~
dense_4079/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_namedense_4079/kernel
w
%dense_4079/kernel/Read/ReadVariableOpReadVariableOpdense_4079/kernel*
_output_shapes

: *
dtype0
v
dense_4078/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namedense_4078/bias
o
#dense_4078/bias/Read/ReadVariableOpReadVariableOpdense_4078/bias*
_output_shapes
: *
dtype0

dense_4078/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *"
shared_namedense_4078/kernel
x
%dense_4078/kernel/Read/ReadVariableOpReadVariableOpdense_4078/kernel*
_output_shapes
:	� *
dtype0
w
dense_4077/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_namedense_4077/bias
p
#dense_4077/bias/Read/ReadVariableOpReadVariableOpdense_4077/bias*
_output_shapes	
:�*
dtype0
�
dense_4077/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*"
shared_namedense_4077/kernel
y
%dense_4077/kernel/Read/ReadVariableOpReadVariableOpdense_4077/kernel* 
_output_shapes
:
��*
dtype0
w
dense_4076/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_namedense_4076/bias
p
#dense_4076/bias/Read/ReadVariableOpReadVariableOpdense_4076/bias*
_output_shapes	
:�*
dtype0

dense_4076/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*"
shared_namedense_4076/kernel
x
%dense_4076/kernel/Read/ReadVariableOpReadVariableOpdense_4076/kernel*
_output_shapes
:	@�*
dtype0
v
dense_4075/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_namedense_4075/bias
o
#dense_4075/bias/Read/ReadVariableOpReadVariableOpdense_4075/bias*
_output_shapes
:@*
dtype0
~
dense_4075/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*"
shared_namedense_4075/kernel
w
%dense_4075/kernel/Read/ReadVariableOpReadVariableOpdense_4075/kernel*
_output_shapes

:@*
dtype0
�
 serving_default_dense_4075_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCall serving_default_dense_4075_inputdense_4075/kerneldense_4075/biasdense_4076/kerneldense_4076/biasdense_4077/kerneldense_4077/biasdense_4078/kerneldense_4078/biasdense_4079/kerneldense_4079/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_28895084

NoOpNoOp
�=
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�<
value�<B�< B�<
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias*
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias*
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias*
J
0
1
2
3
%4
&5
-6
.7
58
69*
J
0
1
2
3
%4
&5
-6
.7
58
69*
* 
�
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
<trace_0
=trace_1
>trace_2
?trace_3* 
6
@trace_0
Atrace_1
Btrace_2
Ctrace_3* 
* 
�
D
_variables
E_iterations
F_learning_rate
G_index_dict
H
_momentums
I_velocities
J_update_step_xla*

Kserving_default* 

0
1*

0
1*
* 
�
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Qtrace_0* 

Rtrace_0* 
a[
VARIABLE_VALUEdense_4075/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_4075/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Xtrace_0* 

Ytrace_0* 
a[
VARIABLE_VALUEdense_4076/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_4076/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

%0
&1*

%0
&1*
* 
�
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

_trace_0* 

`trace_0* 
a[
VARIABLE_VALUEdense_4077/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_4077/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

-0
.1*

-0
.1*
* 
�
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

ftrace_0* 

gtrace_0* 
a[
VARIABLE_VALUEdense_4078/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_4078/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

50
61*

50
61*
* 
�
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

mtrace_0* 

ntrace_0* 
a[
VARIABLE_VALUEdense_4079/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_4079/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

o0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
E0
p1
q2
r3
s4
t5
u6
v7
w8
x9
y10
z11
{12
|13
}14
~15
16
�17
�18
�19
�20*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
L
p0
r1
t2
v3
x4
z5
|6
~7
�8
�9*
L
q0
s1
u2
w3
y4
{5
}6
7
�8
�9*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
c]
VARIABLE_VALUEAdam/m/dense_4075/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/dense_4075/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_4075/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_4075/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/dense_4076/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/dense_4076/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_4076/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_4076/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/dense_4077/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/dense_4077/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_4077/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_4077/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/dense_4078/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/dense_4078/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_4078/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_4078/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/dense_4079/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/dense_4079/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_4079/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_4079/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_4075/kerneldense_4075/biasdense_4076/kerneldense_4076/biasdense_4077/kerneldense_4077/biasdense_4078/kerneldense_4078/biasdense_4079/kerneldense_4079/bias	iterationlearning_rateAdam/m/dense_4075/kernelAdam/v/dense_4075/kernelAdam/m/dense_4075/biasAdam/v/dense_4075/biasAdam/m/dense_4076/kernelAdam/v/dense_4076/kernelAdam/m/dense_4076/biasAdam/v/dense_4076/biasAdam/m/dense_4077/kernelAdam/v/dense_4077/kernelAdam/m/dense_4077/biasAdam/v/dense_4077/biasAdam/m/dense_4078/kernelAdam/v/dense_4078/kernelAdam/m/dense_4078/biasAdam/v/dense_4078/biasAdam/m/dense_4079/kernelAdam/v/dense_4079/kernelAdam/m/dense_4079/biasAdam/v/dense_4079/biastotalcountConst*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_save_28895536
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_4075/kerneldense_4075/biasdense_4076/kerneldense_4076/biasdense_4077/kerneldense_4077/biasdense_4078/kerneldense_4078/biasdense_4079/kerneldense_4079/bias	iterationlearning_rateAdam/m/dense_4075/kernelAdam/v/dense_4075/kernelAdam/m/dense_4075/biasAdam/v/dense_4075/biasAdam/m/dense_4076/kernelAdam/v/dense_4076/kernelAdam/m/dense_4076/biasAdam/v/dense_4076/biasAdam/m/dense_4077/kernelAdam/v/dense_4077/kernelAdam/m/dense_4077/biasAdam/v/dense_4077/biasAdam/m/dense_4078/kernelAdam/v/dense_4078/kernelAdam/m/dense_4078/biasAdam/v/dense_4078/biasAdam/m/dense_4079/kernelAdam/v/dense_4079/kernelAdam/m/dense_4079/biasAdam/v/dense_4079/biastotalcount*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__traced_restore_28895648��
�
�
-__inference_dense_4077_layer_call_fn_28895259

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4077_layer_call_and_return_conditional_losses_28894777p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
L__inference_sequential_815_layer_call_and_return_conditional_losses_28894846
dense_4075_input%
dense_4075_28894820:@!
dense_4075_28894822:@&
dense_4076_28894825:	@�"
dense_4076_28894827:	�'
dense_4077_28894830:
��"
dense_4077_28894832:	�&
dense_4078_28894835:	� !
dense_4078_28894837: %
dense_4079_28894840: !
dense_4079_28894842:
identity��"dense_4075/StatefulPartitionedCall�"dense_4076/StatefulPartitionedCall�"dense_4077/StatefulPartitionedCall�"dense_4078/StatefulPartitionedCall�"dense_4079/StatefulPartitionedCall�
"dense_4075/StatefulPartitionedCallStatefulPartitionedCalldense_4075_inputdense_4075_28894820dense_4075_28894822*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4075_layer_call_and_return_conditional_losses_28894743�
"dense_4076/StatefulPartitionedCallStatefulPartitionedCall+dense_4075/StatefulPartitionedCall:output:0dense_4076_28894825dense_4076_28894827*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4076_layer_call_and_return_conditional_losses_28894760�
"dense_4077/StatefulPartitionedCallStatefulPartitionedCall+dense_4076/StatefulPartitionedCall:output:0dense_4077_28894830dense_4077_28894832*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4077_layer_call_and_return_conditional_losses_28894777�
"dense_4078/StatefulPartitionedCallStatefulPartitionedCall+dense_4077/StatefulPartitionedCall:output:0dense_4078_28894835dense_4078_28894837*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4078_layer_call_and_return_conditional_losses_28894794�
"dense_4079/StatefulPartitionedCallStatefulPartitionedCall+dense_4078/StatefulPartitionedCall:output:0dense_4079_28894840dense_4079_28894842*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4079_layer_call_and_return_conditional_losses_28894810z
IdentityIdentity+dense_4079/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^dense_4075/StatefulPartitionedCall#^dense_4076/StatefulPartitionedCall#^dense_4077/StatefulPartitionedCall#^dense_4078/StatefulPartitionedCall#^dense_4079/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2H
"dense_4075/StatefulPartitionedCall"dense_4075/StatefulPartitionedCall2H
"dense_4076/StatefulPartitionedCall"dense_4076/StatefulPartitionedCall2H
"dense_4077/StatefulPartitionedCall"dense_4077/StatefulPartitionedCall2H
"dense_4078/StatefulPartitionedCall"dense_4078/StatefulPartitionedCall2H
"dense_4079/StatefulPartitionedCall"dense_4079/StatefulPartitionedCall:Y U
'
_output_shapes
:���������
*
_user_specified_namedense_4075_input
�

�
H__inference_dense_4078_layer_call_and_return_conditional_losses_28895290

inputs1
matmul_readvariableop_resource:	� -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
H__inference_dense_4076_layer_call_and_return_conditional_losses_28894760

inputs1
matmul_readvariableop_resource:	@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
L__inference_sequential_815_layer_call_and_return_conditional_losses_28894817
dense_4075_input%
dense_4075_28894744:@!
dense_4075_28894746:@&
dense_4076_28894761:	@�"
dense_4076_28894763:	�'
dense_4077_28894778:
��"
dense_4077_28894780:	�&
dense_4078_28894795:	� !
dense_4078_28894797: %
dense_4079_28894811: !
dense_4079_28894813:
identity��"dense_4075/StatefulPartitionedCall�"dense_4076/StatefulPartitionedCall�"dense_4077/StatefulPartitionedCall�"dense_4078/StatefulPartitionedCall�"dense_4079/StatefulPartitionedCall�
"dense_4075/StatefulPartitionedCallStatefulPartitionedCalldense_4075_inputdense_4075_28894744dense_4075_28894746*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4075_layer_call_and_return_conditional_losses_28894743�
"dense_4076/StatefulPartitionedCallStatefulPartitionedCall+dense_4075/StatefulPartitionedCall:output:0dense_4076_28894761dense_4076_28894763*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4076_layer_call_and_return_conditional_losses_28894760�
"dense_4077/StatefulPartitionedCallStatefulPartitionedCall+dense_4076/StatefulPartitionedCall:output:0dense_4077_28894778dense_4077_28894780*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4077_layer_call_and_return_conditional_losses_28894777�
"dense_4078/StatefulPartitionedCallStatefulPartitionedCall+dense_4077/StatefulPartitionedCall:output:0dense_4078_28894795dense_4078_28894797*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4078_layer_call_and_return_conditional_losses_28894794�
"dense_4079/StatefulPartitionedCallStatefulPartitionedCall+dense_4078/StatefulPartitionedCall:output:0dense_4079_28894811dense_4079_28894813*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4079_layer_call_and_return_conditional_losses_28894810z
IdentityIdentity+dense_4079/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^dense_4075/StatefulPartitionedCall#^dense_4076/StatefulPartitionedCall#^dense_4077/StatefulPartitionedCall#^dense_4078/StatefulPartitionedCall#^dense_4079/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2H
"dense_4075/StatefulPartitionedCall"dense_4075/StatefulPartitionedCall2H
"dense_4076/StatefulPartitionedCall"dense_4076/StatefulPartitionedCall2H
"dense_4077/StatefulPartitionedCall"dense_4077/StatefulPartitionedCall2H
"dense_4078/StatefulPartitionedCall"dense_4078/StatefulPartitionedCall2H
"dense_4079/StatefulPartitionedCall"dense_4079/StatefulPartitionedCall:Y U
'
_output_shapes
:���������
*
_user_specified_namedense_4075_input
�

�
1__inference_sequential_815_layer_call_fn_28894901
dense_4075_input
unknown:@
	unknown_0:@
	unknown_1:	@�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	� 
	unknown_6: 
	unknown_7: 
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_4075_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_815_layer_call_and_return_conditional_losses_28894878o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:���������
*
_user_specified_namedense_4075_input
�

�
1__inference_sequential_815_layer_call_fn_28895134

inputs
unknown:@
	unknown_0:@
	unknown_1:	@�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	� 
	unknown_6: 
	unknown_7: 
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_815_layer_call_and_return_conditional_losses_28894932o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
H__inference_dense_4075_layer_call_and_return_conditional_losses_28895230

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
H__inference_dense_4077_layer_call_and_return_conditional_losses_28894777

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
!__inference__traced_save_28895536
file_prefix:
(read_disablecopyonread_dense_4075_kernel:@6
(read_1_disablecopyonread_dense_4075_bias:@=
*read_2_disablecopyonread_dense_4076_kernel:	@�7
(read_3_disablecopyonread_dense_4076_bias:	�>
*read_4_disablecopyonread_dense_4077_kernel:
��7
(read_5_disablecopyonread_dense_4077_bias:	�=
*read_6_disablecopyonread_dense_4078_kernel:	� 6
(read_7_disablecopyonread_dense_4078_bias: <
*read_8_disablecopyonread_dense_4079_kernel: 6
(read_9_disablecopyonread_dense_4079_bias:-
#read_10_disablecopyonread_iteration:	 1
'read_11_disablecopyonread_learning_rate: D
2read_12_disablecopyonread_adam_m_dense_4075_kernel:@D
2read_13_disablecopyonread_adam_v_dense_4075_kernel:@>
0read_14_disablecopyonread_adam_m_dense_4075_bias:@>
0read_15_disablecopyonread_adam_v_dense_4075_bias:@E
2read_16_disablecopyonread_adam_m_dense_4076_kernel:	@�E
2read_17_disablecopyonread_adam_v_dense_4076_kernel:	@�?
0read_18_disablecopyonread_adam_m_dense_4076_bias:	�?
0read_19_disablecopyonread_adam_v_dense_4076_bias:	�F
2read_20_disablecopyonread_adam_m_dense_4077_kernel:
��F
2read_21_disablecopyonread_adam_v_dense_4077_kernel:
��?
0read_22_disablecopyonread_adam_m_dense_4077_bias:	�?
0read_23_disablecopyonread_adam_v_dense_4077_bias:	�E
2read_24_disablecopyonread_adam_m_dense_4078_kernel:	� E
2read_25_disablecopyonread_adam_v_dense_4078_kernel:	� >
0read_26_disablecopyonread_adam_m_dense_4078_bias: >
0read_27_disablecopyonread_adam_v_dense_4078_bias: D
2read_28_disablecopyonread_adam_m_dense_4079_kernel: D
2read_29_disablecopyonread_adam_v_dense_4079_kernel: >
0read_30_disablecopyonread_adam_m_dense_4079_bias:>
0read_31_disablecopyonread_adam_v_dense_4079_bias:)
read_32_disablecopyonread_total: )
read_33_disablecopyonread_count: 
savev2_const
identity_69��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: z
Read/DisableCopyOnReadDisableCopyOnRead(read_disablecopyonread_dense_4075_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp(read_disablecopyonread_dense_4075_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:@|
Read_1/DisableCopyOnReadDisableCopyOnRead(read_1_disablecopyonread_dense_4075_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp(read_1_disablecopyonread_dense_4075_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_2/DisableCopyOnReadDisableCopyOnRead*read_2_disablecopyonread_dense_4076_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp*read_2_disablecopyonread_dense_4076_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	@�*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	@�d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	@�|
Read_3/DisableCopyOnReadDisableCopyOnRead(read_3_disablecopyonread_dense_4076_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp(read_3_disablecopyonread_dense_4076_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_4/DisableCopyOnReadDisableCopyOnRead*read_4_disablecopyonread_dense_4077_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp*read_4_disablecopyonread_dense_4077_kernel^Read_4/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_5/DisableCopyOnReadDisableCopyOnRead(read_5_disablecopyonread_dense_4077_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp(read_5_disablecopyonread_dense_4077_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_6/DisableCopyOnReadDisableCopyOnRead*read_6_disablecopyonread_dense_4078_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp*read_6_disablecopyonread_dense_4078_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	� *
dtype0o
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	� f
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	� |
Read_7/DisableCopyOnReadDisableCopyOnRead(read_7_disablecopyonread_dense_4078_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp(read_7_disablecopyonread_dense_4078_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: ~
Read_8/DisableCopyOnReadDisableCopyOnRead*read_8_disablecopyonread_dense_4079_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp*read_8_disablecopyonread_dense_4079_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

: |
Read_9/DisableCopyOnReadDisableCopyOnRead(read_9_disablecopyonread_dense_4079_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp(read_9_disablecopyonread_dense_4079_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_10/DisableCopyOnReadDisableCopyOnRead#read_10_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp#read_10_disablecopyonread_iteration^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_11/DisableCopyOnReadDisableCopyOnRead'read_11_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp'read_11_disablecopyonread_learning_rate^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_12/DisableCopyOnReadDisableCopyOnRead2read_12_disablecopyonread_adam_m_dense_4075_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp2read_12_disablecopyonread_adam_m_dense_4075_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_13/DisableCopyOnReadDisableCopyOnRead2read_13_disablecopyonread_adam_v_dense_4075_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp2read_13_disablecopyonread_adam_v_dense_4075_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_14/DisableCopyOnReadDisableCopyOnRead0read_14_disablecopyonread_adam_m_dense_4075_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp0read_14_disablecopyonread_adam_m_dense_4075_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_15/DisableCopyOnReadDisableCopyOnRead0read_15_disablecopyonread_adam_v_dense_4075_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp0read_15_disablecopyonread_adam_v_dense_4075_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_16/DisableCopyOnReadDisableCopyOnRead2read_16_disablecopyonread_adam_m_dense_4076_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp2read_16_disablecopyonread_adam_m_dense_4076_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	@�*
dtype0p
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	@�f
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:	@��
Read_17/DisableCopyOnReadDisableCopyOnRead2read_17_disablecopyonread_adam_v_dense_4076_kernel"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp2read_17_disablecopyonread_adam_v_dense_4076_kernel^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	@�*
dtype0p
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	@�f
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:	@��
Read_18/DisableCopyOnReadDisableCopyOnRead0read_18_disablecopyonread_adam_m_dense_4076_bias"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp0read_18_disablecopyonread_adam_m_dense_4076_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_19/DisableCopyOnReadDisableCopyOnRead0read_19_disablecopyonread_adam_v_dense_4076_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp0read_19_disablecopyonread_adam_v_dense_4076_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_20/DisableCopyOnReadDisableCopyOnRead2read_20_disablecopyonread_adam_m_dense_4077_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp2read_20_disablecopyonread_adam_m_dense_4077_kernel^Read_20/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_21/DisableCopyOnReadDisableCopyOnRead2read_21_disablecopyonread_adam_v_dense_4077_kernel"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp2read_21_disablecopyonread_adam_v_dense_4077_kernel^Read_21/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_22/DisableCopyOnReadDisableCopyOnRead0read_22_disablecopyonread_adam_m_dense_4077_bias"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp0read_22_disablecopyonread_adam_m_dense_4077_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_23/DisableCopyOnReadDisableCopyOnRead0read_23_disablecopyonread_adam_v_dense_4077_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp0read_23_disablecopyonread_adam_v_dense_4077_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_24/DisableCopyOnReadDisableCopyOnRead2read_24_disablecopyonread_adam_m_dense_4078_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp2read_24_disablecopyonread_adam_m_dense_4078_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	� *
dtype0p
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	� f
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:	� �
Read_25/DisableCopyOnReadDisableCopyOnRead2read_25_disablecopyonread_adam_v_dense_4078_kernel"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp2read_25_disablecopyonread_adam_v_dense_4078_kernel^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	� *
dtype0p
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	� f
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:	� �
Read_26/DisableCopyOnReadDisableCopyOnRead0read_26_disablecopyonread_adam_m_dense_4078_bias"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp0read_26_disablecopyonread_adam_m_dense_4078_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_27/DisableCopyOnReadDisableCopyOnRead0read_27_disablecopyonread_adam_v_dense_4078_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp0read_27_disablecopyonread_adam_v_dense_4078_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_28/DisableCopyOnReadDisableCopyOnRead2read_28_disablecopyonread_adam_m_dense_4079_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp2read_28_disablecopyonread_adam_m_dense_4079_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_29/DisableCopyOnReadDisableCopyOnRead2read_29_disablecopyonread_adam_v_dense_4079_kernel"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp2read_29_disablecopyonread_adam_v_dense_4079_kernel^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_30/DisableCopyOnReadDisableCopyOnRead0read_30_disablecopyonread_adam_m_dense_4079_bias"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp0read_30_disablecopyonread_adam_m_dense_4079_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_31/DisableCopyOnReadDisableCopyOnRead0read_31_disablecopyonread_adam_v_dense_4079_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp0read_31_disablecopyonread_adam_v_dense_4079_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_32/DisableCopyOnReadDisableCopyOnReadread_32_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOpread_32_disablecopyonread_total^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_33/DisableCopyOnReadDisableCopyOnReadread_33_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOpread_33_disablecopyonread_count^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*�
value�B�#B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *1
dtypes'
%2#	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_68Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_69IdentityIdentity_68:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_69Identity_69:output:0*[
_input_shapesJ
H: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:#

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
-__inference_dense_4076_layer_call_fn_28895239

inputs
unknown:	@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4076_layer_call_and_return_conditional_losses_28894760p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
H__inference_dense_4078_layer_call_and_return_conditional_losses_28894794

inputs1
matmul_readvariableop_resource:	� -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
H__inference_dense_4079_layer_call_and_return_conditional_losses_28895309

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
1__inference_sequential_815_layer_call_fn_28895109

inputs
unknown:@
	unknown_0:@
	unknown_1:	@�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	� 
	unknown_6: 
	unknown_7: 
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_815_layer_call_and_return_conditional_losses_28894878o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
H__inference_dense_4075_layer_call_and_return_conditional_losses_28894743

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
L__inference_sequential_815_layer_call_and_return_conditional_losses_28894878

inputs%
dense_4075_28894852:@!
dense_4075_28894854:@&
dense_4076_28894857:	@�"
dense_4076_28894859:	�'
dense_4077_28894862:
��"
dense_4077_28894864:	�&
dense_4078_28894867:	� !
dense_4078_28894869: %
dense_4079_28894872: !
dense_4079_28894874:
identity��"dense_4075/StatefulPartitionedCall�"dense_4076/StatefulPartitionedCall�"dense_4077/StatefulPartitionedCall�"dense_4078/StatefulPartitionedCall�"dense_4079/StatefulPartitionedCall�
"dense_4075/StatefulPartitionedCallStatefulPartitionedCallinputsdense_4075_28894852dense_4075_28894854*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4075_layer_call_and_return_conditional_losses_28894743�
"dense_4076/StatefulPartitionedCallStatefulPartitionedCall+dense_4075/StatefulPartitionedCall:output:0dense_4076_28894857dense_4076_28894859*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4076_layer_call_and_return_conditional_losses_28894760�
"dense_4077/StatefulPartitionedCallStatefulPartitionedCall+dense_4076/StatefulPartitionedCall:output:0dense_4077_28894862dense_4077_28894864*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4077_layer_call_and_return_conditional_losses_28894777�
"dense_4078/StatefulPartitionedCallStatefulPartitionedCall+dense_4077/StatefulPartitionedCall:output:0dense_4078_28894867dense_4078_28894869*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4078_layer_call_and_return_conditional_losses_28894794�
"dense_4079/StatefulPartitionedCallStatefulPartitionedCall+dense_4078/StatefulPartitionedCall:output:0dense_4079_28894872dense_4079_28894874*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4079_layer_call_and_return_conditional_losses_28894810z
IdentityIdentity+dense_4079/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^dense_4075/StatefulPartitionedCall#^dense_4076/StatefulPartitionedCall#^dense_4077/StatefulPartitionedCall#^dense_4078/StatefulPartitionedCall#^dense_4079/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2H
"dense_4075/StatefulPartitionedCall"dense_4075/StatefulPartitionedCall2H
"dense_4076/StatefulPartitionedCall"dense_4076/StatefulPartitionedCall2H
"dense_4077/StatefulPartitionedCall"dense_4077/StatefulPartitionedCall2H
"dense_4078/StatefulPartitionedCall"dense_4078/StatefulPartitionedCall2H
"dense_4079/StatefulPartitionedCall"dense_4079/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
L__inference_sequential_815_layer_call_and_return_conditional_losses_28894932

inputs%
dense_4075_28894906:@!
dense_4075_28894908:@&
dense_4076_28894911:	@�"
dense_4076_28894913:	�'
dense_4077_28894916:
��"
dense_4077_28894918:	�&
dense_4078_28894921:	� !
dense_4078_28894923: %
dense_4079_28894926: !
dense_4079_28894928:
identity��"dense_4075/StatefulPartitionedCall�"dense_4076/StatefulPartitionedCall�"dense_4077/StatefulPartitionedCall�"dense_4078/StatefulPartitionedCall�"dense_4079/StatefulPartitionedCall�
"dense_4075/StatefulPartitionedCallStatefulPartitionedCallinputsdense_4075_28894906dense_4075_28894908*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4075_layer_call_and_return_conditional_losses_28894743�
"dense_4076/StatefulPartitionedCallStatefulPartitionedCall+dense_4075/StatefulPartitionedCall:output:0dense_4076_28894911dense_4076_28894913*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4076_layer_call_and_return_conditional_losses_28894760�
"dense_4077/StatefulPartitionedCallStatefulPartitionedCall+dense_4076/StatefulPartitionedCall:output:0dense_4077_28894916dense_4077_28894918*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4077_layer_call_and_return_conditional_losses_28894777�
"dense_4078/StatefulPartitionedCallStatefulPartitionedCall+dense_4077/StatefulPartitionedCall:output:0dense_4078_28894921dense_4078_28894923*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4078_layer_call_and_return_conditional_losses_28894794�
"dense_4079/StatefulPartitionedCallStatefulPartitionedCall+dense_4078/StatefulPartitionedCall:output:0dense_4079_28894926dense_4079_28894928*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4079_layer_call_and_return_conditional_losses_28894810z
IdentityIdentity+dense_4079/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^dense_4075/StatefulPartitionedCall#^dense_4076/StatefulPartitionedCall#^dense_4077/StatefulPartitionedCall#^dense_4078/StatefulPartitionedCall#^dense_4079/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2H
"dense_4075/StatefulPartitionedCall"dense_4075/StatefulPartitionedCall2H
"dense_4076/StatefulPartitionedCall"dense_4076/StatefulPartitionedCall2H
"dense_4077/StatefulPartitionedCall"dense_4077/StatefulPartitionedCall2H
"dense_4078/StatefulPartitionedCall"dense_4078/StatefulPartitionedCall2H
"dense_4079/StatefulPartitionedCall"dense_4079/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_dense_4078_layer_call_fn_28895279

inputs
unknown:	� 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4078_layer_call_and_return_conditional_losses_28894794o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
$__inference__traced_restore_28895648
file_prefix4
"assignvariableop_dense_4075_kernel:@0
"assignvariableop_1_dense_4075_bias:@7
$assignvariableop_2_dense_4076_kernel:	@�1
"assignvariableop_3_dense_4076_bias:	�8
$assignvariableop_4_dense_4077_kernel:
��1
"assignvariableop_5_dense_4077_bias:	�7
$assignvariableop_6_dense_4078_kernel:	� 0
"assignvariableop_7_dense_4078_bias: 6
$assignvariableop_8_dense_4079_kernel: 0
"assignvariableop_9_dense_4079_bias:'
assignvariableop_10_iteration:	 +
!assignvariableop_11_learning_rate: >
,assignvariableop_12_adam_m_dense_4075_kernel:@>
,assignvariableop_13_adam_v_dense_4075_kernel:@8
*assignvariableop_14_adam_m_dense_4075_bias:@8
*assignvariableop_15_adam_v_dense_4075_bias:@?
,assignvariableop_16_adam_m_dense_4076_kernel:	@�?
,assignvariableop_17_adam_v_dense_4076_kernel:	@�9
*assignvariableop_18_adam_m_dense_4076_bias:	�9
*assignvariableop_19_adam_v_dense_4076_bias:	�@
,assignvariableop_20_adam_m_dense_4077_kernel:
��@
,assignvariableop_21_adam_v_dense_4077_kernel:
��9
*assignvariableop_22_adam_m_dense_4077_bias:	�9
*assignvariableop_23_adam_v_dense_4077_bias:	�?
,assignvariableop_24_adam_m_dense_4078_kernel:	� ?
,assignvariableop_25_adam_v_dense_4078_kernel:	� 8
*assignvariableop_26_adam_m_dense_4078_bias: 8
*assignvariableop_27_adam_v_dense_4078_bias: >
,assignvariableop_28_adam_m_dense_4079_kernel: >
,assignvariableop_29_adam_v_dense_4079_kernel: 8
*assignvariableop_30_adam_m_dense_4079_bias:8
*assignvariableop_31_adam_v_dense_4079_bias:#
assignvariableop_32_total: #
assignvariableop_33_count: 
identity_35��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*�
value�B�#B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::*1
dtypes'
%2#	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp"assignvariableop_dense_4075_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_4075_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp$assignvariableop_2_dense_4076_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_4076_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp$assignvariableop_4_dense_4077_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_4077_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp$assignvariableop_6_dense_4078_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_4078_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp$assignvariableop_8_dense_4079_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp"assignvariableop_9_dense_4079_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_iterationIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp,assignvariableop_12_adam_m_dense_4075_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp,assignvariableop_13_adam_v_dense_4075_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_m_dense_4075_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_v_dense_4075_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp,assignvariableop_16_adam_m_dense_4076_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp,assignvariableop_17_adam_v_dense_4076_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_m_dense_4076_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_v_dense_4076_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp,assignvariableop_20_adam_m_dense_4077_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_v_dense_4077_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_m_dense_4077_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_v_dense_4077_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp,assignvariableop_24_adam_m_dense_4078_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp,assignvariableop_25_adam_v_dense_4078_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_m_dense_4078_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_v_dense_4078_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp,assignvariableop_28_adam_m_dense_4079_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp,assignvariableop_29_adam_v_dense_4079_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_m_dense_4079_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_v_dense_4079_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpassignvariableop_32_totalIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpassignvariableop_33_countIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_35IdentityIdentity_34:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_35Identity_35:output:0*Y
_input_shapesH
F: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�:
�

#__inference__wrapped_model_28894728
dense_4075_inputJ
8sequential_815_dense_4075_matmul_readvariableop_resource:@G
9sequential_815_dense_4075_biasadd_readvariableop_resource:@K
8sequential_815_dense_4076_matmul_readvariableop_resource:	@�H
9sequential_815_dense_4076_biasadd_readvariableop_resource:	�L
8sequential_815_dense_4077_matmul_readvariableop_resource:
��H
9sequential_815_dense_4077_biasadd_readvariableop_resource:	�K
8sequential_815_dense_4078_matmul_readvariableop_resource:	� G
9sequential_815_dense_4078_biasadd_readvariableop_resource: J
8sequential_815_dense_4079_matmul_readvariableop_resource: G
9sequential_815_dense_4079_biasadd_readvariableop_resource:
identity��0sequential_815/dense_4075/BiasAdd/ReadVariableOp�/sequential_815/dense_4075/MatMul/ReadVariableOp�0sequential_815/dense_4076/BiasAdd/ReadVariableOp�/sequential_815/dense_4076/MatMul/ReadVariableOp�0sequential_815/dense_4077/BiasAdd/ReadVariableOp�/sequential_815/dense_4077/MatMul/ReadVariableOp�0sequential_815/dense_4078/BiasAdd/ReadVariableOp�/sequential_815/dense_4078/MatMul/ReadVariableOp�0sequential_815/dense_4079/BiasAdd/ReadVariableOp�/sequential_815/dense_4079/MatMul/ReadVariableOp�
/sequential_815/dense_4075/MatMul/ReadVariableOpReadVariableOp8sequential_815_dense_4075_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
 sequential_815/dense_4075/MatMulMatMuldense_4075_input7sequential_815/dense_4075/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0sequential_815/dense_4075/BiasAdd/ReadVariableOpReadVariableOp9sequential_815_dense_4075_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
!sequential_815/dense_4075/BiasAddBiasAdd*sequential_815/dense_4075/MatMul:product:08sequential_815/dense_4075/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
sequential_815/dense_4075/ReluRelu*sequential_815/dense_4075/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
/sequential_815/dense_4076/MatMul/ReadVariableOpReadVariableOp8sequential_815_dense_4076_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype0�
 sequential_815/dense_4076/MatMulMatMul,sequential_815/dense_4075/Relu:activations:07sequential_815/dense_4076/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
0sequential_815/dense_4076/BiasAdd/ReadVariableOpReadVariableOp9sequential_815_dense_4076_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!sequential_815/dense_4076/BiasAddBiasAdd*sequential_815/dense_4076/MatMul:product:08sequential_815/dense_4076/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
sequential_815/dense_4076/ReluRelu*sequential_815/dense_4076/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
/sequential_815/dense_4077/MatMul/ReadVariableOpReadVariableOp8sequential_815_dense_4077_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
 sequential_815/dense_4077/MatMulMatMul,sequential_815/dense_4076/Relu:activations:07sequential_815/dense_4077/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
0sequential_815/dense_4077/BiasAdd/ReadVariableOpReadVariableOp9sequential_815_dense_4077_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!sequential_815/dense_4077/BiasAddBiasAdd*sequential_815/dense_4077/MatMul:product:08sequential_815/dense_4077/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
sequential_815/dense_4077/ReluRelu*sequential_815/dense_4077/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
/sequential_815/dense_4078/MatMul/ReadVariableOpReadVariableOp8sequential_815_dense_4078_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
 sequential_815/dense_4078/MatMulMatMul,sequential_815/dense_4077/Relu:activations:07sequential_815/dense_4078/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
0sequential_815/dense_4078/BiasAdd/ReadVariableOpReadVariableOp9sequential_815_dense_4078_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
!sequential_815/dense_4078/BiasAddBiasAdd*sequential_815/dense_4078/MatMul:product:08sequential_815/dense_4078/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
sequential_815/dense_4078/ReluRelu*sequential_815/dense_4078/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
/sequential_815/dense_4079/MatMul/ReadVariableOpReadVariableOp8sequential_815_dense_4079_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
 sequential_815/dense_4079/MatMulMatMul,sequential_815/dense_4078/Relu:activations:07sequential_815/dense_4079/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
0sequential_815/dense_4079/BiasAdd/ReadVariableOpReadVariableOp9sequential_815_dense_4079_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!sequential_815/dense_4079/BiasAddBiasAdd*sequential_815/dense_4079/MatMul:product:08sequential_815/dense_4079/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������y
IdentityIdentity*sequential_815/dense_4079/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp1^sequential_815/dense_4075/BiasAdd/ReadVariableOp0^sequential_815/dense_4075/MatMul/ReadVariableOp1^sequential_815/dense_4076/BiasAdd/ReadVariableOp0^sequential_815/dense_4076/MatMul/ReadVariableOp1^sequential_815/dense_4077/BiasAdd/ReadVariableOp0^sequential_815/dense_4077/MatMul/ReadVariableOp1^sequential_815/dense_4078/BiasAdd/ReadVariableOp0^sequential_815/dense_4078/MatMul/ReadVariableOp1^sequential_815/dense_4079/BiasAdd/ReadVariableOp0^sequential_815/dense_4079/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2d
0sequential_815/dense_4075/BiasAdd/ReadVariableOp0sequential_815/dense_4075/BiasAdd/ReadVariableOp2b
/sequential_815/dense_4075/MatMul/ReadVariableOp/sequential_815/dense_4075/MatMul/ReadVariableOp2d
0sequential_815/dense_4076/BiasAdd/ReadVariableOp0sequential_815/dense_4076/BiasAdd/ReadVariableOp2b
/sequential_815/dense_4076/MatMul/ReadVariableOp/sequential_815/dense_4076/MatMul/ReadVariableOp2d
0sequential_815/dense_4077/BiasAdd/ReadVariableOp0sequential_815/dense_4077/BiasAdd/ReadVariableOp2b
/sequential_815/dense_4077/MatMul/ReadVariableOp/sequential_815/dense_4077/MatMul/ReadVariableOp2d
0sequential_815/dense_4078/BiasAdd/ReadVariableOp0sequential_815/dense_4078/BiasAdd/ReadVariableOp2b
/sequential_815/dense_4078/MatMul/ReadVariableOp/sequential_815/dense_4078/MatMul/ReadVariableOp2d
0sequential_815/dense_4079/BiasAdd/ReadVariableOp0sequential_815/dense_4079/BiasAdd/ReadVariableOp2b
/sequential_815/dense_4079/MatMul/ReadVariableOp/sequential_815/dense_4079/MatMul/ReadVariableOp:Y U
'
_output_shapes
:���������
*
_user_specified_namedense_4075_input
�-
�
L__inference_sequential_815_layer_call_and_return_conditional_losses_28895172

inputs;
)dense_4075_matmul_readvariableop_resource:@8
*dense_4075_biasadd_readvariableop_resource:@<
)dense_4076_matmul_readvariableop_resource:	@�9
*dense_4076_biasadd_readvariableop_resource:	�=
)dense_4077_matmul_readvariableop_resource:
��9
*dense_4077_biasadd_readvariableop_resource:	�<
)dense_4078_matmul_readvariableop_resource:	� 8
*dense_4078_biasadd_readvariableop_resource: ;
)dense_4079_matmul_readvariableop_resource: 8
*dense_4079_biasadd_readvariableop_resource:
identity��!dense_4075/BiasAdd/ReadVariableOp� dense_4075/MatMul/ReadVariableOp�!dense_4076/BiasAdd/ReadVariableOp� dense_4076/MatMul/ReadVariableOp�!dense_4077/BiasAdd/ReadVariableOp� dense_4077/MatMul/ReadVariableOp�!dense_4078/BiasAdd/ReadVariableOp� dense_4078/MatMul/ReadVariableOp�!dense_4079/BiasAdd/ReadVariableOp� dense_4079/MatMul/ReadVariableOp�
 dense_4075/MatMul/ReadVariableOpReadVariableOp)dense_4075_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_4075/MatMulMatMulinputs(dense_4075/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
!dense_4075/BiasAdd/ReadVariableOpReadVariableOp*dense_4075_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_4075/BiasAddBiasAdddense_4075/MatMul:product:0)dense_4075/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@f
dense_4075/ReluReludense_4075/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
 dense_4076/MatMul/ReadVariableOpReadVariableOp)dense_4076_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype0�
dense_4076/MatMulMatMuldense_4075/Relu:activations:0(dense_4076/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
!dense_4076/BiasAdd/ReadVariableOpReadVariableOp*dense_4076_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_4076/BiasAddBiasAdddense_4076/MatMul:product:0)dense_4076/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������g
dense_4076/ReluReludense_4076/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
 dense_4077/MatMul/ReadVariableOpReadVariableOp)dense_4077_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_4077/MatMulMatMuldense_4076/Relu:activations:0(dense_4077/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
!dense_4077/BiasAdd/ReadVariableOpReadVariableOp*dense_4077_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_4077/BiasAddBiasAdddense_4077/MatMul:product:0)dense_4077/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������g
dense_4077/ReluReludense_4077/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
 dense_4078/MatMul/ReadVariableOpReadVariableOp)dense_4078_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
dense_4078/MatMulMatMuldense_4077/Relu:activations:0(dense_4078/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
!dense_4078/BiasAdd/ReadVariableOpReadVariableOp*dense_4078_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_4078/BiasAddBiasAdddense_4078/MatMul:product:0)dense_4078/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� f
dense_4078/ReluReludense_4078/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
 dense_4079/MatMul/ReadVariableOpReadVariableOp)dense_4079_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_4079/MatMulMatMuldense_4078/Relu:activations:0(dense_4079/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_4079/BiasAdd/ReadVariableOpReadVariableOp*dense_4079_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_4079/BiasAddBiasAdddense_4079/MatMul:product:0)dense_4079/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������j
IdentityIdentitydense_4079/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_4075/BiasAdd/ReadVariableOp!^dense_4075/MatMul/ReadVariableOp"^dense_4076/BiasAdd/ReadVariableOp!^dense_4076/MatMul/ReadVariableOp"^dense_4077/BiasAdd/ReadVariableOp!^dense_4077/MatMul/ReadVariableOp"^dense_4078/BiasAdd/ReadVariableOp!^dense_4078/MatMul/ReadVariableOp"^dense_4079/BiasAdd/ReadVariableOp!^dense_4079/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2F
!dense_4075/BiasAdd/ReadVariableOp!dense_4075/BiasAdd/ReadVariableOp2D
 dense_4075/MatMul/ReadVariableOp dense_4075/MatMul/ReadVariableOp2F
!dense_4076/BiasAdd/ReadVariableOp!dense_4076/BiasAdd/ReadVariableOp2D
 dense_4076/MatMul/ReadVariableOp dense_4076/MatMul/ReadVariableOp2F
!dense_4077/BiasAdd/ReadVariableOp!dense_4077/BiasAdd/ReadVariableOp2D
 dense_4077/MatMul/ReadVariableOp dense_4077/MatMul/ReadVariableOp2F
!dense_4078/BiasAdd/ReadVariableOp!dense_4078/BiasAdd/ReadVariableOp2D
 dense_4078/MatMul/ReadVariableOp dense_4078/MatMul/ReadVariableOp2F
!dense_4079/BiasAdd/ReadVariableOp!dense_4079/BiasAdd/ReadVariableOp2D
 dense_4079/MatMul/ReadVariableOp dense_4079/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_dense_4079_layer_call_fn_28895299

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4079_layer_call_and_return_conditional_losses_28894810o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
H__inference_dense_4079_layer_call_and_return_conditional_losses_28894810

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
-__inference_dense_4075_layer_call_fn_28895219

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_4075_layer_call_and_return_conditional_losses_28894743o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
1__inference_sequential_815_layer_call_fn_28894955
dense_4075_input
unknown:@
	unknown_0:@
	unknown_1:	@�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	� 
	unknown_6: 
	unknown_7: 
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_4075_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_815_layer_call_and_return_conditional_losses_28894932o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:���������
*
_user_specified_namedense_4075_input
�

�
H__inference_dense_4076_layer_call_and_return_conditional_losses_28895250

inputs1
matmul_readvariableop_resource:	@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
&__inference_signature_wrapper_28895084
dense_4075_input
unknown:@
	unknown_0:@
	unknown_1:	@�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	� 
	unknown_6: 
	unknown_7: 
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_4075_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_28894728o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:���������
*
_user_specified_namedense_4075_input
�-
�
L__inference_sequential_815_layer_call_and_return_conditional_losses_28895210

inputs;
)dense_4075_matmul_readvariableop_resource:@8
*dense_4075_biasadd_readvariableop_resource:@<
)dense_4076_matmul_readvariableop_resource:	@�9
*dense_4076_biasadd_readvariableop_resource:	�=
)dense_4077_matmul_readvariableop_resource:
��9
*dense_4077_biasadd_readvariableop_resource:	�<
)dense_4078_matmul_readvariableop_resource:	� 8
*dense_4078_biasadd_readvariableop_resource: ;
)dense_4079_matmul_readvariableop_resource: 8
*dense_4079_biasadd_readvariableop_resource:
identity��!dense_4075/BiasAdd/ReadVariableOp� dense_4075/MatMul/ReadVariableOp�!dense_4076/BiasAdd/ReadVariableOp� dense_4076/MatMul/ReadVariableOp�!dense_4077/BiasAdd/ReadVariableOp� dense_4077/MatMul/ReadVariableOp�!dense_4078/BiasAdd/ReadVariableOp� dense_4078/MatMul/ReadVariableOp�!dense_4079/BiasAdd/ReadVariableOp� dense_4079/MatMul/ReadVariableOp�
 dense_4075/MatMul/ReadVariableOpReadVariableOp)dense_4075_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_4075/MatMulMatMulinputs(dense_4075/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
!dense_4075/BiasAdd/ReadVariableOpReadVariableOp*dense_4075_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_4075/BiasAddBiasAdddense_4075/MatMul:product:0)dense_4075/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@f
dense_4075/ReluReludense_4075/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
 dense_4076/MatMul/ReadVariableOpReadVariableOp)dense_4076_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype0�
dense_4076/MatMulMatMuldense_4075/Relu:activations:0(dense_4076/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
!dense_4076/BiasAdd/ReadVariableOpReadVariableOp*dense_4076_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_4076/BiasAddBiasAdddense_4076/MatMul:product:0)dense_4076/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������g
dense_4076/ReluReludense_4076/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
 dense_4077/MatMul/ReadVariableOpReadVariableOp)dense_4077_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_4077/MatMulMatMuldense_4076/Relu:activations:0(dense_4077/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
!dense_4077/BiasAdd/ReadVariableOpReadVariableOp*dense_4077_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_4077/BiasAddBiasAdddense_4077/MatMul:product:0)dense_4077/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������g
dense_4077/ReluReludense_4077/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
 dense_4078/MatMul/ReadVariableOpReadVariableOp)dense_4078_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
dense_4078/MatMulMatMuldense_4077/Relu:activations:0(dense_4078/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
!dense_4078/BiasAdd/ReadVariableOpReadVariableOp*dense_4078_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_4078/BiasAddBiasAdddense_4078/MatMul:product:0)dense_4078/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� f
dense_4078/ReluReludense_4078/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
 dense_4079/MatMul/ReadVariableOpReadVariableOp)dense_4079_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_4079/MatMulMatMuldense_4078/Relu:activations:0(dense_4079/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_4079/BiasAdd/ReadVariableOpReadVariableOp*dense_4079_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_4079/BiasAddBiasAdddense_4079/MatMul:product:0)dense_4079/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������j
IdentityIdentitydense_4079/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_4075/BiasAdd/ReadVariableOp!^dense_4075/MatMul/ReadVariableOp"^dense_4076/BiasAdd/ReadVariableOp!^dense_4076/MatMul/ReadVariableOp"^dense_4077/BiasAdd/ReadVariableOp!^dense_4077/MatMul/ReadVariableOp"^dense_4078/BiasAdd/ReadVariableOp!^dense_4078/MatMul/ReadVariableOp"^dense_4079/BiasAdd/ReadVariableOp!^dense_4079/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2F
!dense_4075/BiasAdd/ReadVariableOp!dense_4075/BiasAdd/ReadVariableOp2D
 dense_4075/MatMul/ReadVariableOp dense_4075/MatMul/ReadVariableOp2F
!dense_4076/BiasAdd/ReadVariableOp!dense_4076/BiasAdd/ReadVariableOp2D
 dense_4076/MatMul/ReadVariableOp dense_4076/MatMul/ReadVariableOp2F
!dense_4077/BiasAdd/ReadVariableOp!dense_4077/BiasAdd/ReadVariableOp2D
 dense_4077/MatMul/ReadVariableOp dense_4077/MatMul/ReadVariableOp2F
!dense_4078/BiasAdd/ReadVariableOp!dense_4078/BiasAdd/ReadVariableOp2D
 dense_4078/MatMul/ReadVariableOp dense_4078/MatMul/ReadVariableOp2F
!dense_4079/BiasAdd/ReadVariableOp!dense_4079/BiasAdd/ReadVariableOp2D
 dense_4079/MatMul/ReadVariableOp dense_4079/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
H__inference_dense_4077_layer_call_and_return_conditional_losses_28895270

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
M
dense_4075_input9
"serving_default_dense_4075_input:0���������>

dense_40790
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias"
_tf_keras_layer
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias"
_tf_keras_layer
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias"
_tf_keras_layer
f
0
1
2
3
%4
&5
-6
.7
58
69"
trackable_list_wrapper
f
0
1
2
3
%4
&5
-6
.7
58
69"
trackable_list_wrapper
 "
trackable_list_wrapper
�
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
<trace_0
=trace_1
>trace_2
?trace_32�
1__inference_sequential_815_layer_call_fn_28894901
1__inference_sequential_815_layer_call_fn_28894955
1__inference_sequential_815_layer_call_fn_28895109
1__inference_sequential_815_layer_call_fn_28895134�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z<trace_0z=trace_1z>trace_2z?trace_3
�
@trace_0
Atrace_1
Btrace_2
Ctrace_32�
L__inference_sequential_815_layer_call_and_return_conditional_losses_28894817
L__inference_sequential_815_layer_call_and_return_conditional_losses_28894846
L__inference_sequential_815_layer_call_and_return_conditional_losses_28895172
L__inference_sequential_815_layer_call_and_return_conditional_losses_28895210�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z@trace_0zAtrace_1zBtrace_2zCtrace_3
�B�
#__inference__wrapped_model_28894728dense_4075_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
D
_variables
E_iterations
F_learning_rate
G_index_dict
H
_momentums
I_velocities
J_update_step_xla"
experimentalOptimizer
,
Kserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Qtrace_02�
-__inference_dense_4075_layer_call_fn_28895219�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zQtrace_0
�
Rtrace_02�
H__inference_dense_4075_layer_call_and_return_conditional_losses_28895230�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zRtrace_0
#:!@2dense_4075/kernel
:@2dense_4075/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Xtrace_02�
-__inference_dense_4076_layer_call_fn_28895239�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zXtrace_0
�
Ytrace_02�
H__inference_dense_4076_layer_call_and_return_conditional_losses_28895250�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zYtrace_0
$:"	@�2dense_4076/kernel
:�2dense_4076/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�
_trace_02�
-__inference_dense_4077_layer_call_fn_28895259�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z_trace_0
�
`trace_02�
H__inference_dense_4077_layer_call_and_return_conditional_losses_28895270�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z`trace_0
%:#
��2dense_4077/kernel
:�2dense_4077/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
�
ftrace_02�
-__inference_dense_4078_layer_call_fn_28895279�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zftrace_0
�
gtrace_02�
H__inference_dense_4078_layer_call_and_return_conditional_losses_28895290�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zgtrace_0
$:"	� 2dense_4078/kernel
: 2dense_4078/bias
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
�
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
�
mtrace_02�
-__inference_dense_4079_layer_call_fn_28895299�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zmtrace_0
�
ntrace_02�
H__inference_dense_4079_layer_call_and_return_conditional_losses_28895309�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zntrace_0
#:! 2dense_4079/kernel
:2dense_4079/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
'
o0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_sequential_815_layer_call_fn_28894901dense_4075_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_sequential_815_layer_call_fn_28894955dense_4075_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_sequential_815_layer_call_fn_28895109inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_sequential_815_layer_call_fn_28895134inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_sequential_815_layer_call_and_return_conditional_losses_28894817dense_4075_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_sequential_815_layer_call_and_return_conditional_losses_28894846dense_4075_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_sequential_815_layer_call_and_return_conditional_losses_28895172inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_sequential_815_layer_call_and_return_conditional_losses_28895210inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
E0
p1
q2
r3
s4
t5
u6
v7
w8
x9
y10
z11
{12
|13
}14
~15
16
�17
�18
�19
�20"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
h
p0
r1
t2
v3
x4
z5
|6
~7
�8
�9"
trackable_list_wrapper
h
q0
s1
u2
w3
y4
{5
}6
7
�8
�9"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
&__inference_signature_wrapper_28895084dense_4075_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_dense_4075_layer_call_fn_28895219inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dense_4075_layer_call_and_return_conditional_losses_28895230inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_dense_4076_layer_call_fn_28895239inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dense_4076_layer_call_and_return_conditional_losses_28895250inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_dense_4077_layer_call_fn_28895259inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dense_4077_layer_call_and_return_conditional_losses_28895270inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_dense_4078_layer_call_fn_28895279inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dense_4078_layer_call_and_return_conditional_losses_28895290inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_dense_4079_layer_call_fn_28895299inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dense_4079_layer_call_and_return_conditional_losses_28895309inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
(:&@2Adam/m/dense_4075/kernel
(:&@2Adam/v/dense_4075/kernel
": @2Adam/m/dense_4075/bias
": @2Adam/v/dense_4075/bias
):'	@�2Adam/m/dense_4076/kernel
):'	@�2Adam/v/dense_4076/kernel
#:!�2Adam/m/dense_4076/bias
#:!�2Adam/v/dense_4076/bias
*:(
��2Adam/m/dense_4077/kernel
*:(
��2Adam/v/dense_4077/kernel
#:!�2Adam/m/dense_4077/bias
#:!�2Adam/v/dense_4077/bias
):'	� 2Adam/m/dense_4078/kernel
):'	� 2Adam/v/dense_4078/kernel
":  2Adam/m/dense_4078/bias
":  2Adam/v/dense_4078/bias
(:& 2Adam/m/dense_4079/kernel
(:& 2Adam/v/dense_4079/kernel
": 2Adam/m/dense_4079/bias
": 2Adam/v/dense_4079/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count�
#__inference__wrapped_model_28894728�
%&-.569�6
/�,
*�'
dense_4075_input���������
� "7�4
2

dense_4079$�!

dense_4079����������
H__inference_dense_4075_layer_call_and_return_conditional_losses_28895230c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������@
� �
-__inference_dense_4075_layer_call_fn_28895219X/�,
%�"
 �
inputs���������
� "!�
unknown���������@�
H__inference_dense_4076_layer_call_and_return_conditional_losses_28895250d/�,
%�"
 �
inputs���������@
� "-�*
#� 
tensor_0����������
� �
-__inference_dense_4076_layer_call_fn_28895239Y/�,
%�"
 �
inputs���������@
� ""�
unknown�����������
H__inference_dense_4077_layer_call_and_return_conditional_losses_28895270e%&0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
-__inference_dense_4077_layer_call_fn_28895259Z%&0�-
&�#
!�
inputs����������
� ""�
unknown�����������
H__inference_dense_4078_layer_call_and_return_conditional_losses_28895290d-.0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0��������� 
� �
-__inference_dense_4078_layer_call_fn_28895279Y-.0�-
&�#
!�
inputs����������
� "!�
unknown��������� �
H__inference_dense_4079_layer_call_and_return_conditional_losses_28895309c56/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������
� �
-__inference_dense_4079_layer_call_fn_28895299X56/�,
%�"
 �
inputs��������� 
� "!�
unknown����������
L__inference_sequential_815_layer_call_and_return_conditional_losses_28894817}
%&-.56A�>
7�4
*�'
dense_4075_input���������
p

 
� ",�)
"�
tensor_0���������
� �
L__inference_sequential_815_layer_call_and_return_conditional_losses_28894846}
%&-.56A�>
7�4
*�'
dense_4075_input���������
p 

 
� ",�)
"�
tensor_0���������
� �
L__inference_sequential_815_layer_call_and_return_conditional_losses_28895172s
%&-.567�4
-�*
 �
inputs���������
p

 
� ",�)
"�
tensor_0���������
� �
L__inference_sequential_815_layer_call_and_return_conditional_losses_28895210s
%&-.567�4
-�*
 �
inputs���������
p 

 
� ",�)
"�
tensor_0���������
� �
1__inference_sequential_815_layer_call_fn_28894901r
%&-.56A�>
7�4
*�'
dense_4075_input���������
p

 
� "!�
unknown����������
1__inference_sequential_815_layer_call_fn_28894955r
%&-.56A�>
7�4
*�'
dense_4075_input���������
p 

 
� "!�
unknown����������
1__inference_sequential_815_layer_call_fn_28895109h
%&-.567�4
-�*
 �
inputs���������
p

 
� "!�
unknown����������
1__inference_sequential_815_layer_call_fn_28895134h
%&-.567�4
-�*
 �
inputs���������
p 

 
� "!�
unknown����������
&__inference_signature_wrapper_28895084�
%&-.56M�J
� 
C�@
>
dense_4075_input*�'
dense_4075_input���������"7�4
2

dense_4079$�!

dense_4079���������