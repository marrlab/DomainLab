from domainlab.tasks import get_task


def test_task_gettter():
    task = get_task("mini_vlcs")
    print(task)
