import tensorflow as tf

"""
Some utilities for training
"""

class FunctionHook(tf.train.SessionRunHook):
    """
    Turn a Function into a hook. Useful for embedding non-tensorflow logic
    into a MonitoredTrainingSession with the loop context. I use it to write
    progress files.
    
    ```python
    obj = NonTensorFlowEsqueObject()
    @FunctionHook(freq=100)
    def mytask(ctx,run_values):
        print("Hello from you session!")
        stepnum = global_step.eval(session=ctx.session)
        value = value.eval(session=ctx.session)
        obj.DoSomething(value,stepnum)
        
    session = tf.train.MonitoredTrainingSession(
            checkpoint_dir=training_dir,
            hooks=[mytask])
    with session as sess:
        while not sess.should_stop():
            sess.run(ae.train_step)
    ```
    """
    def __init__(self, freq=500):
        self.freq = freq
        self.global_step = tf.train.get_or_create_global_step()
    def after_run(self,run_context,run_values):
        stepnum = run_context.session.run(self.global_step)
        if stepnum%self.freq==self.freq-1:
            self.func(run_context,run_values)
    def __call__(self, func):
        self.func = func
        return self
